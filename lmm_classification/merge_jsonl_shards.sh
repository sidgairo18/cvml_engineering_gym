#!/usr/bin/env bash
set -euo pipefail

IN_DIR="outputs/shards"
OUT_FILE="outputs/val2017_llava7b_two_gated_merged.jsonl"

cat ${IN_DIR}/val2017_llava7b_two_gated_shard_*.jsonl > "${OUT_FILE}"
echo "Merged -> ${OUT_FILE}"

