#!/bin/bash
set -euo pipefail

if command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
elif command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

download_file() {
    local url="$1"
    local output="$2"
    local tmp_output="${output}.part"

    if [[ -s "$output" ]]; then
        echo "Checkpoint already exists: $output"
        return 0
    fi

    rm -f "$tmp_output"
    if [[ "$DOWNLOADER" == "wget" ]]; then
        wget -O "$tmp_output" "$url"
    else
        curl -fL "$url" -o "$tmp_output"
    fi
    mv "$tmp_output" "$output"
}

SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

echo "Downloading sam2.1_hiera_tiny.pt checkpoint..."
download_file "${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt" "sam2.1_hiera_tiny.pt"

echo "Downloading sam2.1_hiera_small.pt checkpoint..."
download_file "${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt" "sam2.1_hiera_small.pt"

echo "Downloading sam2.1_hiera_base_plus.pt checkpoint..."
download_file "${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt" "sam2.1_hiera_base_plus.pt"

echo "Downloading sam2.1_hiera_large.pt checkpoint..."
download_file "${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt" "sam2.1_hiera_large.pt"

echo "All checkpoints are downloaded successfully."
