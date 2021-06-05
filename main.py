"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import warnings
import sys

import torch

# get configs
from config import gridworld_configs, molecule_configs
from config.default import DEFAULT_CONFIG
from config import debug_configs
from metalearner import MetaLearner
from utils.helpers import merge_dicts


def main():
    if len(sys.argv) >= 2: config_name = sys.argv[1]
    else: config_name = "DEFAULT"
    
    if config_name == "DEFAULT":
        config = DEFAULT_CONFIG
    elif config_name == "debug":
        debug_config = getattr(debug_configs, config_name)
        config = merge_dicts(DEFAULT_CONFIG, debug_config)
    else:
        pass
        # config = merge_dicts(DEFAULT_CONFIG, config)

    # --- GridWorld ---

    # standard
    if config["env_name"] == 'gridworld':
        # args = args_gridworld.get_args(rest_args)
        pass
    elif config["env_name"] == 'molecule':
        # args = args_molecule.get_args(rest_args)
        pass
    else:
        name = config["env_name"]
        raise Exception(f"Invalid Environment: {name}")


    # begin training (loop through all passed seeds)
    seed_list = [config["seed"]] if isinstance(config["seed"], int) else config["seed"]
    for seed in seed_list:
        print('training', seed)
        config["seed"] = seed

        learner = MetaLearner(config)
        learner.meta_train()


if __name__ == '__main__':
    main()
