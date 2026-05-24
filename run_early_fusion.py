from MMSA import MMSA_run, get_config_regression

from mmsa_mac_utils import MOSI_ALIGNED_PATH, ensure_file_exists


MODEL_NAME = "ef_lstm"
DATASET_NAME = "mosi"


def main():
    feature_path = ensure_file_exists(MOSI_ALIGNED_PATH)

    config = get_config_regression(MODEL_NAME, DATASET_NAME)
    config["featurePath"] = str(feature_path)
    config["device"] = "cpu"

    print("Experiment: Early Fusion (using ALIGNED data)")
    print("Model:", MODEL_NAME)
    print("Feature file:", feature_path)

    MMSA_run(MODEL_NAME, DATASET_NAME, config=config, seeds=[1111], num_workers=0)


if __name__ == "__main__":
    main()
