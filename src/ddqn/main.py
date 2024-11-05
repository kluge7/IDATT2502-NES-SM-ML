import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_DEBUG"]="INFO"

import torch
print("CUDA available:", torch.cuda.is_available())
print("cuDNN available:",torch.backends.cudnn.is_available())