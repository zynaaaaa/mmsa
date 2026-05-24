from pathlib import Path

import torch
from MMSA import MMSA_run, get_config_regression

original_set_device = torch.cuda.set_device


def safe_set_device(device):
    if str(device) == "cpu" or str(device).startswith("mps"):
        return None
    return original_set_device(device)


torch.cuda.set_device = safe_set_device


def main():
    feature_path = Path(__file__).parent / "MOSI" / "unaligned_50.pkl"

    if not feature_path.exists():
        raise FileNotFoundError(f"找不到特征文件: {feature_path}")

    config = get_config_regression("lmf", "mosi")
    config["featurePath"] = str(feature_path)
    config["device"] = "cpu"

    print("Feature file:", feature_path)
    print("Start running LMF on MOSI with CPU...")

    MMSA_run("lmf", "mosi", config=config, seeds=[1111], num_workers=0)


if __name__ == "__main__":
    main()

