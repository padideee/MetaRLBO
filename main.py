"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import warnings
import sys
import os
import torch

# get configs
from config import AMP_configs
from config.default import DEFAULT_CONFIG
from config import debug_configs
from metalearner import MetaLearner
from utils.helpers import merge_dicts

from random_prog import Program
import utils.helpers as utl

import argparse

parser = argparse.ArgumentParser(description='Process some configs.')
parser.add_argument('--seed', type=int, default=None)

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
    
    args, rest_args = parser.parse_known_args()
    
    if args.seed is not None:
        print("OVERRIDING SEED")
        config["seed"] = args.seed # Override the seed
        config["exp_label"] = config["exp_label"] + "_seed-" + str(args.seed)

    # standard
    if config["task"] == 'AMP-v0' or config["task"] == 'CLAMP-v0':
        pass
    else:
        name = config["task"]
        raise Exception(f"Invalid Task: {name}")


    with open(os.path.join(os.path.abspath(os.getcwd()),'data', 'history.fasta'), 'w') as f:
        pass

    utl.seed(config["seed"])


    if "RANDOM" in config["policy"]["model_name"]:
        prog = Program(config)
        prog.run()
    else:
        metalearner = MetaLearner(config)
        metalearner.run()


if __name__ == '__main__':
    main()
