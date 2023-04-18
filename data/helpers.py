import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch

from collections import OrderedDict

char_pairs = [('A', 0), ('R', 1), ('N', 2), ('D', 3), ('C', 4), ('E', 5), ('Q', 6), ('G', 7), ('H', 8), ('I', 9), ('L', 10), ('K', 11), ('M', 12), ('F', 13), ('P', 14), ('S', 15), ('T', 16), ('W', 17), ('Y', 18), ('V', 19), ('>', 20)]
mol_enc = OrderedDict(char_pairs)
enc_mol = OrderedDict(list(map(lambda x : (x[1], x[0]), char_pairs)))

def seq_to_enc(seq, enc_len=50, num_actions=21):
    enc = [None for i in range(enc_len)]
    for i in range(enc_len):
        if i < len(seq):
            enc[i] = mol_enc[seq[i]]
        else:
            assert 20 < num_actions 
            enc[i] = 20
    
    return F.one_hot(torch.tensor(enc), num_classes=num_actions).numpy()

def enc_to_seq(enc):
    """
        Converts encoding (50, 21) to string...
    """
    enc = torch.argmax(enc, -1)
    seq = []
    for i in range(len(enc)):
        seq.append(enc_mol[enc[i].item()])

    return ''.join(seq)

def enc_to_seq_RNA(enc):
    """
        Converts encoding (14, 4) to string...
    """
    char_pairs_RNA = [('U', 0), ('G', 1), ('C', 2), ('A', 3)]
    mol_enc_RNA = OrderedDict(char_pairs_RNA)
    enc_mol_RNA = OrderedDict(list(map(lambda x: (x[1], x[0]), char_pairs_RNA)))

    enc = torch.argmax(enc, -1)
    seq = []
    for i in range(len(enc)):
        seq.append(enc_mol_RNA[enc[i].item()])

    return ''.join(seq)