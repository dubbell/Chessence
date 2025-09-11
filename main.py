import yaml
import argparse
from train.experiments.train_sac import train_sac


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file to train bot.")

    return parser.parse_args()


# required entries in configuration file
REQ_CONFIG = ["algorithm", "max_timesteps", "checkpoint"]

def main():
    # get run configuration
    args = get_args()
    assert args.config is not None, "Please provide config file."
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
    except:
        raise IOError("Config file not found.")
    

    for req in REQ_CONFIG:
        assert req in config, f"Config missing: `{req}`"
    
    if config["algorithm"].lower() == "sac":
        train_sac(config)
    else:
        raise IOError("Invalid RL algorithm.")


if __name__ == "__main__":
    main()