import argparse

from MMSA import MMSA_run, get_config_regression

from mmsa_mac_utils import MOSI_UNALIGNED_PATH, ensure_file_exists


MODEL_NAME = "self_mm"
DATASET_NAME = "mosi"
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def run_self_mm_multiseed(seeds):
    feature_path = ensure_file_exists(MOSI_UNALIGNED_PATH)

    config = get_config_regression(MODEL_NAME, DATASET_NAME)
    config["featurePath"] = str(feature_path)
    config["device"] = "cpu"

    print("Experiment: Self-MM strong multimodal baseline multi-seed")
    print("Model:", MODEL_NAME)
    print("Feature file:", feature_path)
    print("Seeds:", seeds)
    print("Self-MM 在 CPU 上可能会比较慢。")

    MMSA_run(MODEL_NAME, DATASET_NAME, config=config, seeds=seeds, num_workers=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Self-MM on MOSI with multiple random seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def main():
    args = parse_args()
    run_self_mm_multiseed(args.seeds)


if __name__ == "__main__":
    main()
