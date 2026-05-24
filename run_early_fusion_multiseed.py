import argparse

from MMSA import MMSA_run, get_config_regression

from mmsa_mac_utils import MOSI_ALIGNED_PATH, ensure_file_exists, get_device


MODEL_NAME = "ef_lstm"
DATASET_NAME = "mosi"
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def run_early_fusion_multiseed(seeds):
    feature_path = ensure_file_exists(MOSI_ALIGNED_PATH)

    config = get_config_regression(MODEL_NAME, DATASET_NAME)
    config["featurePath"] = str(feature_path)
    config["device"] = str(get_device())

    print("Experiment: Early Fusion multi-seed (using ALIGNED data)")
    print("Model:", MODEL_NAME)
    print("Feature file:", feature_path)
    print("Seeds:", seeds)

    MMSA_run(MODEL_NAME, DATASET_NAME, config=config, seeds=seeds, num_workers=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Early Fusion on MOSI with multiple random seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def main():
    args = parse_args()
    run_early_fusion_multiseed(args.seeds)


if __name__ == "__main__":
    main()
