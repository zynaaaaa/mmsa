import argparse
from run_mmsa_multiseed import run_mmsa_multiseed


DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def main():
    parser = argparse.ArgumentParser(description="Run LMF on MOSI with multiple random seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()
    run_mmsa_multiseed("lmf", args.seeds)


if __name__ == "__main__":
    main()
