import argparse

from MMSA import MMSA_run, get_config_regression

from mmsa_mac_utils import MOSI_UNALIGNED_PATH, MOSI_ALIGNED_PATH, ensure_file_exists


MODEL_DISPLAY_NAMES = {
    "ef_lstm": "Early Fusion",
    "lf_dnn": "Late Fusion",
    "lmf": "LMF",
    "mult": "MulT",
}
# 需要对齐数据的模型
MODELS_NEED_ALIGNED = {"ef_lstm"}

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DATASET_NAME = "mosi"


def run_mmsa_multiseed(model_name, seeds):
    # 根据模型选择正确的数据文件
    if model_name in MODELS_NEED_ALIGNED:
        feature_path = ensure_file_exists(MOSI_ALIGNED_PATH)
        data_type = "ALIGNED"
    else:
        feature_path = ensure_file_exists(MOSI_UNALIGNED_PATH)
        data_type = "UNALIGNED"

    config = get_config_regression(model_name, DATASET_NAME)
    config["featurePath"] = str(feature_path)
    config["device"] = "cpu"

    print(f"Experiment: {MODEL_DISPLAY_NAMES.get(model_name, model_name)} multi-seed")
    print("Model:", model_name)
    print(f"Feature file: {feature_path} ({data_type})")
    print("Seeds:", seeds)
    if model_name == "mult":
        print("MulT 在 CPU 上会比较慢，3 个 seed 可能需要较长时间。")

    MMSA_run(model_name, DATASET_NAME, config=config, seeds=seeds, num_workers=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MMSA official models with multiple random seeds.")
    parser.add_argument("model", choices=["ef_lstm", "lf_dnn", "lmf", "mult"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def main():
    args = parse_args()
    run_mmsa_multiseed(args.model, args.seeds)


if __name__ == "__main__":
    main()
