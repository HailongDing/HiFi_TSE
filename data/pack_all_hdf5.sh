#!/usr/bin/env bash
# Pack all HDF5 files sequentially, one source at a time.
# Run in tmux to survive connection drops.
#
# Usage: bash data/pack_all_hdf5.sh
#
# Each source is independent — if one fails, the rest still run.
# Already-completed files are NOT skipped (will overwrite).
# Use --source flag on prepare_manifest.py to re-run a single source.

set -uo pipefail

cd /home/hailong/code/from_draft/v1
CONFIG=configs/hifi_tse.yaml
ENV=tse_dataset

SOURCES=(dns4 ears vctk demand wham audioset dns5_noise rir)

echo "=== HDF5 Packing: All Sources ==="
echo "Config: $CONFIG"
echo "Environment: $ENV"
echo "Sources: ${SOURCES[*]}"
echo "Started: $(date)"
echo ""

for src in "${SOURCES[@]}"; do
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] START: $src"
    echo "============================================================"

    conda run -n "$ENV" \
        python data/prepare_manifest.py --config "$CONFIG" --source "$src"

    status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE: $src (success)"
    else
        echo "[$(date '+%H:%M:%S')] FAIL: $src (exit code $status)"
    fi
    echo ""
done

echo "============================================================"
echo "All sources processed. $(date)"
echo ""
echo "=== File sizes ==="
ls -lh /data/tse_hdf5/train/clean_speech/*.h5 2>/dev/null
ls -lh /data/tse_hdf5/train/noise/*.h5 2>/dev/null
ls -lh /data/tse_hdf5/train/rir/*.h5 2>/dev/null
echo ""
du -sh /data/tse_hdf5/train/
