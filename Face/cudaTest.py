import torch

if torch.cuda.is_available():
    print("GPU is available")
    print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("GPU is not available, using CPU")

