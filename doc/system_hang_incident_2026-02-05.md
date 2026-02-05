# System Hang & Reboot Incident Report

- **Date**: Feb 5, 2026
- **Host**: gpu-server (MSI Z790 GAMING PLUS WIFI, Intel i-series, 62 GB RAM, NVIDIA RTX 4090 D)
- **Kernel**: 5.15.0-164-generic (Ubuntu 22.04)
- **Storage**: 2x 1.8TB NVMe (nvme0n1 = root, nvme1n1 = /data)

---

## Summary

The system experienced an unclean reboot at **Feb 5, 2026, 03:21:52 UTC** after being in a degraded state for approximately 13 hours. The root cause was an **Out-of-Memory (OOM) kernel panic** triggered by PyTorch training processes (`pt_main_thread`) that consumed all 62 GB of RAM and all 8 GB of swap.

Because `panic_on_oom` was enabled on this system, the kernel panicked instead of using the OOM killer to terminate the offending process.

---

## Timeline

| Time (UTC)               | Event                                                                                     |
|--------------------------|-------------------------------------------------------------------------------------------|
| Jan 29, 06:15            | System booted cleanly (previous boot)                                                     |
| Feb 4, ~06:08            | User `hailong` SSH sessions active; PyTorch training running with 5 `pt_main_thread` workers |
| Feb 4, 13:17             | Last normal log entry (CRON job). Memory pressure likely critical around this time         |
| Feb 4, 13:17 - 14:14     | **57-minute complete log gap** -- system unresponsive due to NVMe saturated with swap I/O  |
| Feb 4, 14:14             | All 4 cloudflared tunnel connections failed simultaneously (network timeouts)              |
| Feb 4, 14:16             | `snapd` watchdog timeout (5min limit) -- original snapd PID 1554 from boot became unresponsive |
| Feb 4, 14:26 - 14:51     | SIGKILL on snapd process took **25 minutes** to complete -- processes stuck in D state     |
| Feb 4, 15:40             | First `systemd-journald` watchdog timeout (3min limit) -- journald could not write to disk |
| Feb 4, 15:40 - 23:38     | journald repeatedly crashed and restarted (~10+ times), each instance timing out           |
| Feb 4, 23:38             | Journal file corruption detected: `system.journal corrupted or uncleanly shut down`        |
| Feb 5, 03:20             | Last log entries from the dying boot session                                               |
| Feb 5, 03:21:52          | Kernel panic -> system rebooted (kernel uptime ~594330s = ~6.9 days)                       |

---

## Root Cause: OOM Kernel Panic

### Trigger

From the pstore crash dump (`/var/lib/systemd/pstore/177026165/dmesg.txt`):

```
Kernel panic - not syncing: Out of memory: system-wide panic_on_oom is enabled
```

The panic was triggered by `cloudflared` (PID 602154) attempting a page allocation (`GFP_HIGHUSER_MOVABLE`) when the system had zero free swap and insufficient free RAM.

### Memory Consumers

Five `pt_main_thread` (PyTorch DataLoader/training worker) processes consumed virtually all system memory:

| PID    | Process         | RSS (RAM)   | Swap Used  | Total VM      |
|--------|-----------------|-------------|------------|---------------|
| 575349 | pt_main_thread  | ~22.3 GB    | ~4.2 GB    | ~38.7 GB      |
| 575317 | pt_main_thread  | ~18.9 GB    | ~3.6 GB    | ~34.5 GB      |
| 575351 | pt_main_thread  | ~15.8 GB    | ~0.9 GB    | ~28.8 GB      |
| 575350 | pt_main_thread  | ~2.6 GB     | ~0.5 GB    | ~15.2 GB      |
| 575177 | pt_main_thread  | ~0.4 GB     | ~0.7 GB    | ~23.2 GB      |
| **Total** |              | **~60 GB**  | **~10 GB** |               |

### System Memory State at Crash

- **RAM**: 62 GB total, ~582 MB free (below kernel min watermark of 387 MB for Normal zone)
- **Swap**: 8 GB total, **0 bytes free** -- completely exhausted
- **File cache**: ~2 MB remaining (active_file: 357 pages, inactive_file: 157 pages)
- **Swap cache churn**: 4,298,141 swap-in operations -- extreme swap thrashing
- **Anonymous pages**: ~60 GB (active_anon + inactive_anon)

### Why the System Was Degraded for 13 Hours Before the Panic

1. As PyTorch workers consumed more memory, the kernel started heavy swap I/O
2. The NVMe root drive became saturated serving swap pages (4.3 million swap operations)
3. All other disk I/O was starved -- logs couldn't be written, services couldn't start
4. Processes stuck in D state (uninterruptible I/O wait) could not be killed even with SIGKILL
5. Swap took ~13 hours to fill completely (8 GB swap with constant thrashing)
6. Once swap hit 0 bytes free, the next page allocation triggered the OOM -> kernel panic

---

## Evidence

### Key Log Entries

1. **snapd watchdog timeout** (first sign of trouble):
   ```
   Feb 04 14:16:47 systemd[1]: snapd.service: Watchdog timeout (limit 5min)!
   ```

2. **journald watchdog timeouts** (repeated ~10 times):
   ```
   Feb 04 15:40:23 systemd[1]: systemd-journald.service: Watchdog timeout (limit 3min)!
   Feb 04 17:16:59 systemd[1]: systemd-journald.service: Watchdog timeout (limit 3min)!
   Feb 04 18:04:02 systemd[1]: systemd-journald.service: Watchdog timeout (limit 3min)!
   Feb 04 19:43:34 systemd[1]: systemd-journald.service: Watchdog timeout (limit 3min)!
   Feb 04 21:22:16 systemd[1]: systemd-journald.service: Watchdog timeout (limit 3min)!
   Feb 04 21:45:04 systemd[1]: systemd-journald.service: Watchdog timeout (limit 3min)!
   Feb 04 22:32:18 systemd[1]: systemd-journald.service: Watchdog timeout (limit 3min)!
   Feb 04 22:43:45 systemd[1]: systemd-journald.service: Watchdog timeout (limit 3min)!
   ```

3. **Journal corruption** (repeated):
   ```
   systemd-journald: File system.journal corrupted or uncleanly shut down, renaming and replacing.
   ```

4. **OOM killer invoked -> kernel panic** (from pstore):
   ```
   cloudflared invoked oom-killer: gfp_mask=0x1100cca(GFP_HIGHUSER_MOVABLE), order=0
   Kernel panic - not syncing: Out of memory: system-wide panic_on_oom is enabled
   ```

### Reboot Confirmation

```
$ last reboot | head -2
reboot   system boot  5.15.0-164-gener Thu Feb  5 03:21   still running
reboot   system boot  5.15.0-164-gener Thu Jan 29 06:15   still running
```

No matching `shutdown` entry for the Feb 5 reboot -- confirms unclean/crash reboot.

---

## Recommendations

### Immediate

1. **Disable `panic_on_oom`** so the OOM killer can terminate the offending process instead of crashing the entire system:
   ```bash
   sudo sysctl -w vm.panic_on_oom=0
   # Make permanent:
   echo "vm.panic_on_oom=0" | sudo tee -a /etc/sysctl.d/99-oom.conf
   ```

2. **Fix PyTorch training memory usage** -- the 5 `pt_main_thread` workers consumed 60+ GB of RAM. Investigate:
   - DataLoader `num_workers` setting (each worker duplicates dataset in memory)
   - Batch size and accumulation
   - Whether tensors or data are being accumulated across iterations/epochs without being freed
   - Consider using `pin_memory=True` with smaller worker counts

### Preventive

3. **Set memory limits for training jobs** using cgroups or systemd scopes:
   ```bash
   systemd-run --scope -p MemoryMax=50G --user python train.py
   ```

4. **Increase swap size** or add a swap file (current 8 GB is small for 62 GB RAM with heavy workloads):
   ```bash
   sudo fallocate -l 32G /swapfile2
   sudo chmod 600 /swapfile2
   sudo mkswap /swapfile2
   sudo swapon /swapfile2
   ```

5. **Monitor memory during training**:
   ```bash
   watch -n 5 free -h
   ```
   Or set up an alert script that warns when available memory drops below a threshold.

6. **Protect critical services** from OOM killer by adjusting `oom_score_adj`:
   ```bash
   # Protect sshd so you can always log in
   echo -1000 | sudo tee /proc/$(pgrep -x sshd)/oom_score_adj
   ```
   Or permanently via systemd drop-in:
   ```bash
   sudo mkdir -p /etc/systemd/system/ssh.service.d
   echo -e "[Service]\nOOMScoreAdjust=-1000" | sudo tee /etc/systemd/system/ssh.service.d/oom.conf
   sudo systemctl daemon-reload
   ```

---

## Root Cause Analysis: Code-Level Findings

Investigation of the codebase identified the specific code paths responsible for unbounded
memory growth across the 5 `pt_main_thread` processes.

### Finding 1: Unbounded h5py Chunk Cache

**Files:** `data/dataset.py` — `CleanSpeechIndex._get_handle` (line 137),
`NoiseIndex._get_handle` (line 194), `RIRIndex._get_handle` (line 229)

All three HDF5 index classes opened files with default chunk cache settings:
```python
self._handles[path] = h5py.File(path, "r")  # no rdcc_nbytes limit
```

h5py maintains an in-memory raw data chunk cache per file handle. With thousands of
utterances read randomly over 500K training steps, this cache grows unbounded. With
`num_workers=4` DataLoader workers (each forked with independent h5py handles), the total
cache memory across 5 processes scaled to 60GB+.

### Finding 2: HDF5 Handles Never Closed

**File:** `data/dataset.py`

Each index class has a `close()` method (`CleanSpeechIndex.close`, `NoiseIndex.close`,
`RIRIndex.close`) but these methods were **never called** by the training script. The
`HiFiTSEDataset` class had no `close_handles()` method.

### Finding 3: Workers Never Restart

**File:** `train.py` — `infinite_loader()` (line 37-41), `make_train_loader()` (line 44-50)

```python
def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch
```

Combined with `num_workers=4`, DataLoader workers persist for the entire training run
(or until a phase transition). Each worker's h5py chunk caches accumulate without limit.

### Finding 4: No Handle Cleanup at Phase Transitions

**File:** `train.py` (lines 218-231)

At phase transitions, the DataLoader was recreated but old h5py handles were not explicitly
closed. Without `close()` + `gc.collect()`, old handles and their caches lingered in memory.

### Finding 5: No Memory Monitoring

The training loop logged losses and LR but not RSS memory usage. There was no way to detect
memory growth before it reached critical levels.

---

## Fixes Applied (commit TBD)

### Code Fixes

| # | File | Change |
|---|------|--------|
| 1 | `data/dataset.py` | Cap h5py chunk cache: `h5py.File(path, "r", rdcc_nbytes=512*1024)` on all 3 index classes |
| 2 | `data/dataset.py` | Add `HiFiTSEDataset.close_handles()` method that calls `.close()` on all indices |
| 3 | `train.py` | Call `close_handles()` + `gc.collect()` at phase transitions and on resume |
| 4 | `train.py` | Log RSS memory (via `resource.getrusage`) at each `log_interval` step |

### System Fixes (manual, require sudo)

| # | Action | Command |
|---|--------|---------|
| 5 | Disable panic_on_oom | `sudo sysctl -w vm.panic_on_oom=0` + persist to `/etc/sysctl.d/99-oom.conf` |
| 6 | Protect sshd from OOM | systemd drop-in: `OOMScoreAdjust=-1000` for ssh.service |
