import ctypes
import os

# Set the LD_LIBRARY_PATH again just to confirm in the session
os.environ["LD_LIBRARY_PATH"] = "/cluster/home/andreksv/PycharmProjects/IDATT2502-NES-SM-ML/venv/lib64:/cluster/home/andreksv/PycharmProjects/IDATT2502-NES-SM-ML/venv/lib/python3.9/site-packages/nvidia/cusparse/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
print("LD_LIBRARY_PATH:", os.environ["LD_LIBRARY_PATH"])

# Try loading the specific library manually
try:
    ctypes.CDLL("/cluster/home/andreksv/PycharmProjects/IDATT2502-NES-SM-ML/venv/lib64/libcusparse.so.11")
    print("libcusparse.so.11 loaded successfully!")
except OSError as e:
    print("Error loading libcusparse.so.11:", e)

# Check CUDA and cuDNN availability in PyTorch
import torch
print("CUDA available:", torch.cuda.is_available())
print("cuDNN available:", torch.backends.cudnn.is_available())
