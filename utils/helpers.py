from oracles.models import *
from torch.nn import functional as F
import copy
import torch
from torch import nn
import pandas as pd
import os
from collections import OrderedDict
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import data.helpers as data_utl

import json
import random
import numpy as np


def task_str_len(task_name):
    if "AMP" in task_name:
        length = 50
    elif 'AltIsing20' in task_name:
        length=20
    elif 'AltIsing50' in task_name:
        length=50
    elif 'AltIsing100' in task_name:
        length=100
    elif 'RNA14' in task_name:
        length=14
    else:
        raise NotImplementedError
    return length


def mol_to_string_encoding(config, mols):
    """
        Returns: List[String] - List of the mols in string format
    """
    length = task_str_len(config["task"])
    mol_strings = []
    if "AMP" in config["task"]:
        for i in range(len(mols)):
            mol_strings.append(data_utl.enc_to_seq(mols[i]))
    elif "Ising" in config["task"]: 
        for i in range(len(mols)):
            mol_strings.append(data_utl.enc_to_seq(mols[i]))
    elif "RNA14" in config["task"]: # TODO:
        assert NotImplementedError # TODO
    else:
        assert NotImplementedError

    return mol_strings



def save_queried_mols(dataset, folder):
    mols=dataset.mol_strings
    scores=dataset.scores[:dataset.storage_filled].numpy()
    mol_rounds = dataset.mol_round[:dataset.storage_filled].numpy()

    data = {
        "seq": mols,
        'score': scores,
        'mol_round': mol_rounds,
    }
    torch.save(data, os.path.join(folder, 'queried_mols.pt'))

def save_sampled_mols(dataset, folder):
    mols=dataset.mol_strings
    mol_rounds = dataset.mol_round
    mol_query_proxy_idx = dataset.mol_query_proxy_idx

    data = {
        "seq": mols,
        'mol_round': mol_rounds,
        'query_proxy_idx': mol_query_proxy_idx
    }
    torch.save(data, os.path.join(folder, 'sampled_mols.pt'))

def save_config(config, folder):
    with open(os.path.join(folder, 'config.json'), 'w') as fp:
        json.dump(config, fp)


def get_true_oracle_model(config):
    """
        Returns an instance of TrueOracle following config
    """
    if config["true_oracle"]["model_name"] == 'RFC':
        model = RFC(n_estimators = config["true_oracle"]["config"]["n_estimators"])
    elif config["true_oracle"]["model_name"] == 'NN':
        model = NN()
    elif config["true_oracle"]["model_name"] == 'AltIsing_Oracle': # Only for AltIsing Task
        model = AltIsingModel(length=config["task_config"]["seq_len"], vocab_size=20)
    elif config["true_oracle"]["model_name"] == 'RNA14_Oracle': # Only for RNA14 Task
        model = RNA_Model()
    else:
        raise NotImplementedError


    return model


def setup_task_configs(config):
    config["task_config"] = {}
    if config["task"] == "AMP-v0":
        config["task_config"]["seq_len"], config["task_config"]["alphabet_len"] = 50, 21
    elif 'AltIsing' in config["task"]:
        length = task_str_len(config["task"])
        config["task_config"]["seq_len"], config["task_config"]["alphabet_len"] = length, 20
    elif config['task'] == 'RNA14-v0':
        config["task_config"]["seq_len"], config["task_config"]["alphabet_len"] = 14, 4

    else:
        raise NotImplementedError

    return config


def seed(seed, deterministic_execution=False):
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Note that due to parallel processing results will be similar but not identical. '
              'Use only one process and set --deterministic_execution to True if you want identical results '
              '(only recommended for debugging).')

def reset_env(env, config, indices=None, state=None):
    """ env can be many environments or just one """
    # reset all environments
    if indices is not None:
        assert not isinstance(indices[0], bool)
    if (indices is None) or (len(indices) == config["num_processes"]):
        state = torch.tensor(env.reset()).float()
    # reset only the ones given by indices
    else:
        assert state is not None
        for i in indices:
            state[i] = env.reset(index=i)

    return state



# def add_mols_to_history(mols, query_history):
#     """
#         Args:
#             - mols: list of numpy arrays of dim (50, 21)
#             - query_history: list of numpy arrays of dim (50 , 21)
#         Return:
#             - True if mol is not in query_history
#     """

    
def get_proxy_oracle_model(config):
    """
        Returns an instance of (AMP)ProxyOracle following config.
    """

    if config["proxy_oracle"]["model_name"] == 'RFC':
        model = RFC()
    elif config["proxy_oracle"]["model_name"] == 'NN':
        model = NN()
    elif config["proxy_oracle"]["model_name"] == 'RFR':
        model = RFR()
    elif config["proxy_oracle"]["model_name"] == 'KNR':
        model = KNR()
    elif config["proxy_oracle"]["model_name"] == 'RR':
        model = RR()
    elif config["proxy_oracle"]["model_name"] == 'BR':
        model = BR()
    elif config["proxy_oracle"]["model_name"] == 'XGBoost':
        model = XGBoost()
    elif config["proxy_oracle"]["model_name"] == 'XGB':
        model = XGB()
    elif config["proxy_oracle"]["model_name"] == 'GPR':
        model = GPR(kernel = config["proxy_oracle"]["config"]["kernel"])
    elif config["proxy_oracle"]["model_name"] == 'MLP':
        model = MLP(seq_len=config["task_config"]["seq_len"], alphabet_len=config["task_config"]["alphabet_len"])
    elif config["proxy_oracle"]["model_name"] == 'CNN':
        model = CNN(seq_len=config["task_config"]["seq_len"], alphabet_len=config["task_config"]["alphabet_len"])
    else:
        raise NotImplementedError


    return model



def to_one_hot(config, mols):

    if config["task"] == "AMP":

        mols = F.one_hot(mols.long(), num_classes=21).float() # Leo: ISSUE -- the default is 0 if there's no choice of action...?
    else:
        raise NotImplementedError
    return mols

def scores_to_labels(model_name, model, scores):

    if model_name == "RFC":
        labels = scores.argmax(dim=1).unsqueeze(-1)
    else:
        raise NotImplementedError

    return labels



def merge_dicts(d1, d2):
    """
    Args:
        d1 (dict): Dict 1.
        d2 (dict): Dict 2.
    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged




def deep_update(original,
                new_dict,
                new_keys_allowed=False,
                allow_new_subkey_list=None,
                override_all_if_type_changes=None):
    """Updates original dict with values from new_dict recursively.
    If new key is introduced in new_dict, then if new_keys_allowed is not
    True, an error will be thrown. Further, for sub-dicts, if the key is
    in the allow_new_subkey_list, then new subkeys can be introduced.
    Args:
        original (dict): Dictionary with default values.
        new_dict (dict): Dictionary with values to be updated
        new_keys_allowed (bool): Whether new keys are allowed.
        allow_new_subkey_list (Optional[List[str]]): List of keys that
            correspond to dict values where new subkeys can be introduced.
            This is only at the top level.
        override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (dict), iff the "type" key in that value dict changes.
    """
    allow_new_subkey_list = allow_new_subkey_list or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise Exception("Unknown config parameter `{}` ".format(k))

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Allowed key -> ok to add new subkeys.
            elif k in allow_new_subkey_list:
                deep_update(original[k], value, True)
            # Non-allowed key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original



class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return torch.zeros(0, ).to(device)


class convertor:
    def __init__(self):
        self.AA_intg = OrderedDict([('A', 0), ('R', 1), ('N', 2), ('D', 3), ('C', 4), ('E', 5), ('Q', 6), ('G', 7),
                                 ('H', 8), ('I', 9), ('L', 10), ('K', 11), ('M', 12), ('F', 13), ('P', 14), ('S', 15),
                                 ('T', 16), ('W', 17), ('Y', 18), ('V', 19), ('X', 20)])

        self.intg_AA = {v: k for k, v in self.AA_intg.items()}
        # TODO unit-test to make sure padding 'X' does not cause interferece (could be important for future modifications in pipeline)
        # TODO: making sure the character "X" is the default "uncommon Amino Acid" in blast


    def AA_to_one_hot(self):
        # TODO : moving the to_one_hot function in this class
        pass


    def one_hot_to_AA(self, one_hot):
        """ input is the one hot encoded sequence"""
        # tmp = torch.sum(one_hot, dim=(1))
        # tmp = 20 * (1 - tmp)

        intg = torch.argmax(one_hot, dim=-1)
        # import pdb; pdb.set_trace()
        # intg += tmp

        AA_seq = [self.intg_AA[int(i)] for i in intg]

        # my_seq = ''.join(i for i in AA_seq if i not in '>')
        my_seq = ''.join(AA_seq)

        return my_seq


def make_fasta(seq, idx=1):
    """
        input seq is in string format
        This function creates a fasta file for the given sequence
    """
    seq1 = SeqRecord(Seq(seq), id="seq{}".format(idx))
    SeqIO.write(seq1, "data/seq.fasta", "fasta")


def append_history_fasta(seq="data/seq.fasta" , history="data/history.fasta"):
    """
    inputs are the path to the sequence and history in the fasta format
    """
    with open(history, 'a+') as f:
        with open(seq, 'r') as g:
            new_seq = g.read()
            f.write(new_seq)





