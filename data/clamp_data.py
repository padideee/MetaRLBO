import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from storage.query_storage import QueryStorage

from collections import OrderedDict

from data.helpers import seq_to_enc, enc_to_seq

from common_evaluation.clamp_common_eval.defaults import get_default_data_splits


def get_CLAMP_data(mode="test", neg_ratio = 2):

    data = get_default_data_splits(setting='Target') # or get_default_data_splits(setting='Title')
    df = data.get_dataframe() # Get D1 and D2 (D3 is included)
    if mode == "test":
        D1 = data.sample(dataset = "D1", neg_ratio = neg_ratio)
        D2 = data.sample(dataset = "D2", neg_ratio = neg_ratio)

        D = {'AMP': D1['AMP'] + D2['AMP'], 'nonAMP': D1['nonAMP'] + D2['nonAMP']}
    elif mode == "val":
        D = data.sample(dataset = "D1", neg_ratio = neg_ratio)
    else:
        raise NotImplementedError

    # filtered_seq = df.sequence.map(lambda x : 'C' not in x and len(x) >= 15 and len(x) <= 50)
    # df = df[filtered_seq] # Originall cysteine was removed in DynaPPO -- but this is not the case here...

    D['AMP'] = torch.tensor(list(map(lambda x : seq_to_enc(x), D['AMP'])))
    D['nonAMP'] = torch.tensor(list(map(lambda x : seq_to_enc(x), D['nonAMP'])))

    data = torch.cat((D['AMP'], D['nonAMP']), 0)
    print("Size of Data:", data.shape)
    print("Positives:", D['AMP'].shape)
    print("Negatives:", D['nonAMP'].shape)
    scores = [1 for i in range(len(D['AMP']))] + [0 for i in range(len(D['nonAMP']))]

    storage = QueryStorage(data.shape[0], data.shape[1:])

    storage.mols = data

    storage.scores = torch.tensor(scores)

    return storage

