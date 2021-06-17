import numpy as np
import torch
import hickle as hkl
from positional_encodings import PositionalEncoding1D, PositionalEncoding2D


def seq_to_encoding(seq):
    """
    This function takes in a sequence of amino acids, represented numerically,
    and first converts to a 46x21 encoding which is passed to the positional
    encoding function to get a 1D output
    """
    enc = np.zeros((46, 21)) # TODO: Move all these to config
    for i, j in enumerate(seq):
        #print("Here: ", i, j)
        enc[int(i)][int(j)] = 1

    enc = torch.from_numpy(enc.reshape(1,46,21))
    pos_enc = PositionalEncoding1D(46)
    p_enc = pos_enc(enc)

    seqN = enc + p_enc
    # s = (seqN.detach().cpu().numpy()).reshape((1, 46*21))
    return seqN.reshape((1, 46*21))


def get_data(data):

    my_data = hkl.load(data)
    data_x = np.array(my_data['xtrain'])
    data_x = torch.from_numpy(data_x)
    n, x1, x2 = data_x.shape
    # print("Data_x shape: ", n, x1, x2) # (batchsize, x, ch dimension)

    data_y = np.array(my_data['ytrain'])

    pos_enc = PositionalEncoding1D(x1)
    p_enc = pos_enc(data_x)
    # print("PE shape: ", p_enc.shape) # (batchsize, x, ch dimension)

    # mean = np.mean(data_y)

    labels = ['positive']*len(data_y)
    for i in range(data_y.shape[0]):
        if data_y[i] == 4.0:
            labels[i] = 'negative'
    # print(data_x.shape)

    # data_x shape: (batch, 46, 21)
    return data_x + p_enc, labels