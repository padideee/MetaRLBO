"""
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
from data.process_data import get_AMP_data
from oracles.AMP_true_oracle import AMPTrueOracle
import utils.helpers as utl


def main():
    if len(sys.argv) >= 2: config_name = sys.argv[1]
    else: config_name = "DEFAULT"
    
    if config_name == "DEFAULT":
        config = DEFAULT_CONFIG
    elif config_name == "debug":
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


    # The seq and the label from library
    # seq shape: (batch, 46*21)
    # label shape: (batch) -> in binary format:{'positive': AMP, 'negative': not AMP}
    D_AMP = get_AMP_data('data/data.hkl') 

    true_oracle = AMPTrueOracle(training_storage=D_AMP)

    true_oracle_model = utl.get_true_oracle_model(config)


    true_oracle_model = true_oracle.fit(true_oracle_model, flatten_input = True)
    import pdb; pdb.set_trace()

    equiv = (true_oracle.query(true_oracle_model, D_AMP.mols).argmax(1) == D_AMP.scores.squeeze(-1).numpy())

if __name__ == '__main__':
    main()
