#!/usr/bin/env bash
set -e

# Reuse the existing plotting script; point it at the oriented CSV and it will use the PMI column.
python scripts/plot_pmi_top_bottom.py \
  --pmi_csv outputs/coco_co/cooccur_oriented.csv \
  --top_k 20 \
  --out outputs/coco_co/pmi_top_bottom_plot_k20.png

