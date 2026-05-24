import argparse
import pickle

from run_unimodal_multiseed import run_one_seed, summarize, print_summary
from mmsa_mac_utils import MOSI_UNALIGNED_PATH, ensure_file_exists


MODALITY = "vision"
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Visual-only multi-seed experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    return parser.parse_args()


def main():
    args = parse_args()
    feature_path = ensure_file_exists(MOSI_UNALIGNED_PATH)
    print("Experiment: Visual-only multi-seed")
    print("Feature file:", feature_path)
    print("Seeds:", args.seeds)

    with feature_path.open("rb") as f:
        data = pickle.load(f)

    results = [run_one_seed(data, MODALITY, seed) for seed in args.seeds]
    print_summary(summarize(results))


if __name__ == "__main__":
    main()
