from MMSA import MMSA_run, get_config_regression

from mmsa_mac_utils import MOSI_UNALIGNED_PATH, ensure_file_exists, get_device


MODEL_NAME = "lmf"
DATASET_NAME = "mosi"


def main():
    feature_path = ensure_file_exists(MOSI_UNALIGNED_PATH)

    config = get_config_regression(MODEL_NAME, DATASET_NAME)
    config["featurePath"] = str(feature_path)
    config["device"] = str(get_device())

    print("Experiment: LMF")
    print("Model:", MODEL_NAME)
    print("Feature file:", feature_path)

    MMSA_run(MODEL_NAME, DATASET_NAME, config=config, seeds=[1111], num_workers=0)


if __name__ == "__main__":
    main()
