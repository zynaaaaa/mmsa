from run_text_attention_multiseed import run_one_seed
from mmsa_mac_utils import MOSI_UNALIGNED_PATH, ensure_file_exists

import pickle


SEED = 1111


def main():
    feature_path = ensure_file_exists(MOSI_UNALIGNED_PATH)
    print("Experiment: Text-BiLSTM-Attention strong unimodal baseline")
    print("Feature file:", feature_path)
    print("Seed:", SEED)

    with feature_path.open("rb") as f:
        data = pickle.load(f)

    result = run_one_seed(data, SEED)
    print("Test result:", result)


if __name__ == "__main__":
    main()
