from pathlib import Path

import torch
from MMSA import MMSA_run, get_config_regression

from mmsa_mac_utils import get_device


def main():
    device = get_device()
    feature_path = Path(__file__).parent / "MOSI" / "unaligned_50.pkl"

    if not feature_path.exists():
        raise FileNotFoundError(f"找不到特征文件: {feature_path}")

    config = get_config_regression("lmf", "mosi")
    config["featurePath"] = str(feature_path)
    config["device"] = str(device)

    print("Feature file:", feature_path)
    print("Device:", device)
    print("Start running LMF on MOSI...")

    MMSA_run("lmf", "mosi", config=config, seeds=[1111], num_workers=0)


if __name__ == "__main__":
    main()
