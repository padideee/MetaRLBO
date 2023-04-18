'''
    Taken from https://github.com/padideee/MBRL-for-AMP/blob/43a1b3b247faeb28b64eaa06a4d0abdc301fba21/data/process_data.py



    Process data from custom generated data...
'''

import numpy as np
import torch
# import hickle as hkl
import pickle as pkl
from positional_encodings import PositionalEncoding1D, PositionalEncoding2D
import torch.nn.functional as F

def seq_to_encoding(seq):
    """
    This function takes in a sequence of amino acids, represented numerically,
    and first converts to a 46x21 encoding which is passed to the positional
    encoding function to get a 1D output

    Args:
      - seq: (batch_size, T, state dim)
    Return:
      - 
    """
    # enc = np.zeros((46, 21)) # TODO: Move all these to config
    # for i, j in enumerate(seq):
    #     #print("Here: ", i, j)
    #     enc[int(i)][int(j)] = 1

    # enc = torch.from_numpy(enc.reshape(1,46,21))
    # import pdb; pdb.set_trace()
    pos_enc = PositionalEncoding1D(seq.shape[1])
    p_enc = pos_enc(seq)

    seqN = seq + p_enc
    # s = (seqN.detach().cpu().numpy()).reshape((1, 46*21))

    return seqN
    # return seq


def get_data(data):

    # my_data = hkl.load(data)
    # use the pickle format for compute canada..
    with open(data, 'rb') as f:
        my_data = pkl.load(f)
    data_x = np.array(my_data['xtrain'])
    data_x = torch.from_numpy(data_x)
    n, x1, x2 = data_x.shape
    # print("Data_x shape: ", n, x1, x2) # (batchsize, x, ch dimension)

    data_y = np.array(my_data['ytrain'])

    # pos_enc = PositionalEncoding1D(x1)
    # p_enc = pos_enc(data_x)
    # print("PE shape: ", p_enc.shape) # (batchsize, x, ch dimension)

    # mean = np.mean(data_y)

    labels = ['positive']*len(data_y)
    for i in range(data_y.shape[0]):
        if data_y[i] == 4.0:
            labels[i] = 'negative'
    # print(data_x.shape)

    # data_x shape: (batch, 51, 21)
    return data_x, labels




from storage.query_storage import QueryStorage
def get_AMP_data(data_path):

    orig_data, labels = get_data(data_path)


    # Some tricks to get the padding correct for the data...

    # import pdb; pdb.set_trace()


    data = F.one_hot(torch.ones((orig_data.shape[0], 50)).long() * 20, num_classes=21).double()


    data[:, :orig_data.shape[1], :] *= (1 - (orig_data > 0).sum(-1)).unsqueeze(-1)


    data[:, :orig_data.shape[1], :] += orig_data


    # Leo: TODO - pad to 50, 21
    # import pdb; pdb.set_trace()

    labels = torch.tensor([x == 'positive' for x in labels]).long().unsqueeze(-1)


    storage = QueryStorage(data.shape[0], data.shape[1:])
    storage.mols = data
    storage.scores = labels

    return storage


