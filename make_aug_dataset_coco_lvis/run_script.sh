python make_inpainted_dataset.py \
  --dataset coco \
  --images-dir /data/coco/images/train2017 \
  --ann-file /data/coco/annotations/instances_train2017.json \
  --out-dir /data/out/coco_qwen_inpaint \
  --split-name train2017 \
  --strategy sizebased --k-remove 1 --area-max 0.30 --dilate 5 \
  --backend qwen_controlnet \
  --qwen-base Qwen/Qwen-Image \
  --qwen-controlnet InstantX/Qwen-Image-ControlNet-Inpainting \
  --prompt "一张自然的场景，背景干净、无缝填充，与上下文一致" \
  --neg-prompt "模糊、伪影、失真、文字、水印" \
  --sd-steps 30 --qwen-true-cfg-scale 4.0 --qwen-cond-scale 1.0 \
  --dtype bf16 --device cuda

