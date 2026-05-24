from MMSA import MMSA_run, get_config_regression

from mmsa_mac_utils import MOSI_UNALIGNED_PATH, ensure_file_exists


MODEL_NAME = "self_mm"
DATASET_NAME = "mosi"
SEED = 1111


def main():
    feature_path = ensure_file_exists(MOSI_UNALIGNED_PATH)

    config = get_config_regression(MODEL_NAME, DATASET_NAME)
    config["featurePath"] = str(feature_path)
    config["device"] = "cpu"

    print("Experiment: Self-MM strong multimodal baseline")
    print("Model:", MODEL_NAME)
    print("Feature file:", feature_path)
    print("Seed:", SEED)
    print("Self-MM 在 CPU 上可能会比较慢。")

    MMSA_run(MODEL_NAME, DATASET_NAME, config=config, seeds=[SEED], num_workers=0)


if __name__ == "__main__":
    main()
