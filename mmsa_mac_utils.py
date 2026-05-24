from pathlib import Path

import torch


def get_device():
    """自动检测最佳可用设备: CUDA -> MPS -> CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# 仅在非 CUDA 环境下 monkey-patch torch.cuda.set_device，
# 避免 MMSA 框架在 Mac/CPU 环境下调用 cuda.set_device 时报错。
if not torch.cuda.is_available():
    original_set_device = torch.cuda.set_device

    def safe_set_device(device):
        if str(device) == "cpu" or str(device).startswith("mps"):
            return None
        return original_set_device(device)

    torch.cuda.set_device = safe_set_device


PROJECT_DIR = Path(__file__).parent
MOSI_UNALIGNED_PATH = PROJECT_DIR / "MOSI" / "unaligned_50.pkl"
MOSI_ALIGNED_PATH = PROJECT_DIR / "MOSI" / "aligned_50.pkl"


def ensure_file_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    return path
