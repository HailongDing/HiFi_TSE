#!/usr/bin/env bash
# Monitor training log and send notifications to WeChat Work webhook.
# Usage: bash monitor_training.sh <log_file>

set -uo pipefail

LOG_FILE="${1:-./checkpoints/train.log}"
WEBHOOK="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=08365047-6447-4f2c-81d1-6975fd0e1fa0"
HOSTNAME=$(hostname)
LAST_NOTIFY_STEP=-1
NOTIFY_INTERVAL=1000  # send progress every N steps

send_msg() {
    local content="$1"
    curl -s -X POST "$WEBHOOK" \
        -H "Content-Type: application/json" \
        -d "{\"msgtype\":\"markdown\",\"markdown\":{\"content\":\"$content\"}}" \
        > /dev/null 2>&1
}

send_msg "**HiFi-TSE Training Started**
> Host: ${HOSTNAME}
> Config: batch_size=2, grad_accum=32, lr=0.0002
> Total steps: 500,000
> Log: ${LOG_FILE}"

# Wait for log file to appear
while [ ! -f "$LOG_FILE" ]; do
    sleep 2
done

# Tail the log and react to events
tail -n 0 -F "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do

    # Error / OOM detection
    if echo "$line" | grep -qiE "error|exception|traceback|OOM|out of memory|CUDA|RuntimeError|killed"; then
        send_msg "**Training ERROR**
> \`\`\`
> ${line}
> \`\`\`"
    fi

    # Phase transition
    if echo "$line" | grep -q "==> Phase"; then
        send_msg "**Phase Transition**
> ${line}"
    fi

    # Checkpoint saved
    if echo "$line" | grep -q "Saved checkpoint"; then
        send_msg "**Checkpoint Saved**
> ${line}"
    fi

    # Validation result
    if echo "$line" | grep -q "Validation loss"; then
        send_msg "**Validation Result**
> ${line}"
    fi

    # Milestone evaluation
    if echo "$line" | grep -q "MILESTONE_EVAL"; then
        send_msg "**Milestone Evaluation**
> ${line}"
    fi

    # Real audio check
    if echo "$line" | grep -q "REAL_AUDIO_CHECK"; then
        send_msg "**Real Audio Check**
> ${line}"
    fi

    # GAN stability warning
    if echo "$line" | grep -q "GAN_WARNING"; then
        send_msg "**GAN Stability Warning**
> ${line}"
    fi

    # Early warning: possible over-suppression
    if echo "$line" | grep -q "EARLY_CHECK WARNING"; then
        send_msg "**⚠️ EARLY WARNING: Possible Over-Suppression ⚠️**
> ${line}
> Action: Check rms_ratio trend. If sustained < 0.3, consider restarting."
    fi

    # Over-suppression detection from training log rms_ratio
    # Skip TA batches (stft 0.0000 means all-TA batch, not real suppression)
    if echo "$line" | grep -qE "^step.*rms 0\.([0-2][0-9]|0[0-9])" && \
       ! echo "$line" | grep -q "stft 0.0000"; then
        send_msg "**⚠️ Over-Suppression Alert**
> ${line}
> rms_ratio < 0.3 detected in training log (non-TA batch)"
    fi

    # Step progress (every NOTIFY_INTERVAL steps)
    if echo "$line" | grep -qE "^step[[:space:]]+[0-9]"; then
        step=$(echo "$line" | grep -oE "step[[:space:]]+[0-9]+" | grep -oE "[0-9]+")
        if [ -n "$step" ]; then
            # Notify at step 0, then every NOTIFY_INTERVAL steps
            if [ "$step" -eq 0 ] || \
               { [ "$step" -ge $((LAST_NOTIFY_STEP + NOTIFY_INTERVAL)) ] && \
                 [ $((step % NOTIFY_INTERVAL)) -eq 0 ]; }; then
                LAST_NOTIFY_STEP=$step
                # Extract loss values from the line
                send_msg "**Training Progress (step ${step})**
> ${line}"
            fi
        fi
    fi

    # Training complete
    if echo "$line" | grep -q "Training complete"; then
        send_msg "**Training Complete!**
> ${line}"
    fi

done
