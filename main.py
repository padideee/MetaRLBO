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
from config import AMP_configs, CLAMP_configs, Ising20_configs, Ising50_configs, Ising100_configs, RNA14_configs
from config.default import DEFAULT_CONFIG
from config import debug_configs
from metalearner import MetaLearner
from learner import Learner
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
    elif "clamp" in config_name:
        clamp_config = getattr(CLAMP_configs, config_name)
        config = merge_dicts(DEFAULT_CONFIG, clamp_config)
    elif "amp" in config_name:
        amp_config = getattr(AMP_configs, config_name)
        config = merge_dicts(DEFAULT_CONFIG, amp_config)
    elif "ising" in config_name:
        if "ising20" in config_name:
            ising_config = getattr(Ising20_configs, config_name)
        elif "ising50" in config_name:
            ising_config = getattr(Ising50_configs, config_name)
        elif "ising100" in config_name:
            ising_config = getattr(Ising100_configs, config_name)
        else:
            raise NotImplementedError
        config = merge_dicts(DEFAULT_CONFIG, ising_config)
    elif "rna14" in config_name:
        rna_config = getattr(RNA14_configs, config_name)
        config = merge_dicts(DEFAULT_CONFIG, rna_config)
    else:
        raise NotImplementedError

    
    args, rest_args = parser.parse_known_args()
    
    if args.seed is not None:
        print("OVERRIDING SEED")
        config["seed"] = args.seed # Override the seed
        config["exp_label"] = config["exp_label"] + "_seed-" + str(args.seed)

    # standard
    valid_tasks = ['AMP-v0', 'CLAMP-v0', 'AltIsing20-v0', 'AltIsing50-v0', 'AltIsing100-v0', 'RNA14-v0']
    if config["task"] in valid_tasks:
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
    elif not config["use_metalearner"]:
        learner = Learner(config)
        learner.run()
    else:
        metalearner = MetaLearner(config)
        metalearner.run()

        if config["task"] == 'CLAMP-v0':
            from clamp_evaluation import Evaluation

            evaluation = Evaluation(config, metalearner)
            evaluation.run()
        

if __name__ == '__main__':
    main()
