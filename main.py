"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import warnings
import sys

import torch

# get configs
from config import AMP_configs
from config.default import DEFAULT_CONFIG
from config import debug_configs
from metalearner import MetaLearner
from utils.helpers import merge_dicts

from random_prog import Program


def main():
    if len(sys.argv) >= 2: config_name = sys.argv[1]
    else: config_name = "DEFAULT"
    
    if config_name == "DEFAULT":
        config = DEFAULT_CONFIG
    elif "debug" in config_name:
        debug_config = getattr(debug_configs, config_name)
        config = merge_dicts(DEFAULT_CONFIG, debug_config)
    else:
        amp_config = getattr(AMP_configs, config_name)
        config = merge_dicts(DEFAULT_CONFIG, amp_config)


    # standard
    if config["task"] == 'AMP':
        pass
    else:
        name = config["task"]
        raise Exception(f"Invalid Task: {name}")



    if config["policy"]["model_name"] == "RANDOM":
        prog = Program(config)
        prog.run()
    else:
        metalearner = MetaLearner(config)
        metalearner.run()


if __name__ == '__main__':
    main()
