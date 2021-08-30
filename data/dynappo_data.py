import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from storage.query_storage import QueryStorage

from collections import OrderedDict

enc_len = 50
num_actions = 21
mol_enc =   OrderedDict([('A', 0), ('R', 1), ('N', 2), ('D', 3), ('C', 4), ('E', 5), ('Q', 6), ('G', 7), ('H', 8), ('I', 9), ('L', 10), ('K', 11), ('M', 12), ('F', 13), ('P', 14), ('S', 15), ('T', 16), ('W', 17), ('Y', 18), ('V', 19), ('>', 20)])

def seq_to_enc(seq):
    enc = [None for i in range(enc_len)]
    for i in range(enc_len):
        if i < len(seq):
            enc[i] = mol_enc[seq[i]]
        else:
            enc[i] = 20
    
    return F.one_hot(torch.tensor(enc), num_classes=num_actions).numpy()

def get_AMP_data(mode):

    if mode == "val":
        df = pd.read_csv("data/210820_v0.24.1_rf_not-alibicani.csv")
        # df = pd.read_csv("210820_v0.24.1_rf_not-alibicani.csv")
    elif mode == "test":
        df = pd.read_csv("data/210820_v0.24.1_rf_alibicani.csv")
        # df = pd.read_csv("210820_v0.24.1_rf_alibicani.csv")
    else:
        raise NotImplementedError
 
    df.sequence = df.sequence.map(seq_to_enc)

    storage = QueryStorage(df.sequence.shape[0], df.sequence.shape[1:])

    storage.mols = torch.tensor(df.sequence.to_list())

    storage.scores = torch.tensor(df.is_amp.to_list()).long()

    return storage

