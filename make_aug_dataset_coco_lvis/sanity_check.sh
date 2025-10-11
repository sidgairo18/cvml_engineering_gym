python - << 'PY'
import torch
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
from diffusers import QwenImageControlNetModel, QwenImageControlNetInpaintPipeline
print("Diffusers Qwen inpaint OK:", QwenImageControlNetModel is not None)
PY

