from pathlib import Path

import torch

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
