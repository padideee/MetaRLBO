import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from storage.query_storage import QueryStorage

enc_len = 50
num_actions = 21
mol_enc =   {'A': 0,
             'F': 1,
             'R': 2,
             'G': 3,
             'C': 4,
             'W': 5,
             'T': 6,
             'K': 7,
             'N': 8,
             'Y': 9,
             'S': 10,
             'P': 11,
             'L': 12,
             'H': 13,
             'I': 14,
             'Q': 15,
             'V': 16,
             'M': 17,
             'E': 18,
             'D': 19}

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

