from oracles.models import *
from torch.nn import functional as F
import copy
from torch import nn
import pandas as pd
import os
import json
import random
import torch
import numpy as np

def save_mols(mols, scores, folder):
    mols = [mols[i] for i in range(mols.shape[0])]
    data = {
        "seq": mols,
        'pred_prob': scores,
    }
    df = pd.DataFrame(data=data)
    df.to_pickle(os.path.join(folder, 'queried_mols.pkl'))

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
    else:
        raise NotImplementedError


    return model

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