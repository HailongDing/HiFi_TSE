#!/usr/bin/env bash
# Download DNS5 noise_fullband dataset (48kHz, ~62k files, 150 classes)
# Source: DNS Challenge 5 (ICASSP 2023) Azure blob storage
#
# Usage: bash data/download_dns5_noise.sh
# Run in tmux to survive connection drops.

set -euo pipefail

BASE_URL="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/noise_fullband"
DEST_DIR="/data/48khz_dataset/dns_noise_dataset/dns5_noise"
DOWNLOAD_DIR="${DEST_DIR}/_archives"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$DEST_DIR"

ARCHIVES=(
    "datasets_fullband.noise_fullband.audioset_000.tar.bz2"
    "datasets_fullband.noise_fullband.audioset_001.tar.bz2"
    "datasets_fullband.noise_fullband.audioset_002.tar.bz2"
    "datasets_fullband.noise_fullband.audioset_003.tar.bz2"
    "datasets_fullband.noise_fullband.audioset_004.tar.bz2"
    "datasets_fullband.noise_fullband.audioset_005.tar.bz2"
    "datasets_fullband.noise_fullband.audioset_006.tar.bz2"
    "datasets_fullband.noise_fullband.freesound_000.tar.bz2"
    "datasets_fullband.noise_fullband.freesound_001.tar.bz2"
)

echo "=== DNS5 Noise Fullband Download ==="
echo "Destination: $DEST_DIR"
echo "Archives: ${#ARCHIVES[@]} files"
echo ""

# Download function for a single archive
download_one() {
    local archive="$1"
    local url="${BASE_URL}/${archive}"
    local dest="${DOWNLOAD_DIR}/${archive}"

    if [ -f "${dest}.done" ]; then
        echo "[SKIP] ${archive} (already downloaded)"
        return 0
    fi

    echo "[DOWN] ${archive} ..."
    wget -c -q --show-progress -O "$dest" "$url" 2>&1
    touch "${dest}.done"
    echo "[DONE] ${archive}"
}

export -f download_one
export BASE_URL DOWNLOAD_DIR

# Download all archives in parallel (3 at a time to avoid throttling)
printf '%s\n' "${ARCHIVES[@]}" | xargs -P 3 -I {} bash -c 'download_one "$@"' _ {}

echo ""
echo "=== All downloads complete. Starting extraction... ==="
echo ""

# Extract each archive into DEST_DIR
for archive in "${ARCHIVES[@]}"; do
    marker="${DOWNLOAD_DIR}/${archive}.extracted"
    if [ -f "$marker" ]; then
        echo "[SKIP] Extract ${archive} (already extracted)"
        continue
    fi

    echo "[EXTRACT] ${archive} ..."
    tar -xjf "${DOWNLOAD_DIR}/${archive}" -C "$DEST_DIR" --strip-components=2
    touch "$marker"
    echo "[DONE] Extract ${archive}"
done

echo ""
echo "=== Extraction complete ==="

# Summary
total_wav=$(find "$DEST_DIR" -name "*.wav" | wc -l)
total_size=$(du -sh "$DEST_DIR" | cut -f1)
echo "Total WAV files: ${total_wav}"
echo "Total size: ${total_size}"
echo ""

# Clean up archives if all extracted
echo "Archives kept in ${DOWNLOAD_DIR}/ for safety."
echo "Run 'rm -rf ${DOWNLOAD_DIR}' to free ~39 GB after verification."
