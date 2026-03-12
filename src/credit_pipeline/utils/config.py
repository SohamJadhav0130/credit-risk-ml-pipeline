import yaml

from credit_pipeline.utils.paths import CONFIG_DIR


def load_config(config_path=None) -> dict:
    if config_path is None:
        config_path = CONFIG_DIR / "preprocessing_config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
