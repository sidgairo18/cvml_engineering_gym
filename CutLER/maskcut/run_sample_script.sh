#python demo.py --img-path imgs/demo2.jpg \
#  --N 3 --tau 0.15 --vit-arch base --patch-size 8 \
#  --output_path "./test_output_maskcut"

python cutler_cub-200_demo.py \
  --dataset_root ./data \
  --output_path ./outputs_cub200 \
  --samples-per-class 10 \
  --vit-arch small \
  --vit-feat k \
  --patch-size 8 \
  --N 3 \
  --tau 0.15 \
  --fixed_size 480
