import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class MLP(nn.Module): # Remember to put this to device...
    def __init__(self,
                 seq_len=50,
                 alphabet_len=20,
                 n_out = 1, 
                 n_hidden = [32, 8, 4],
                 nonlinearity = F.relu
                 ):
        super().__init__()

        n_layers = tuple(n_hidden) + (n_out, )

        # fully connected layers
        self.fc_layers = nn.ModuleList([nn.LazyLinear(n_layers[i]) for i in range(len(n_layers))])
        self.nonlinearity = nonlinearity

    def forward(self, x):
        for k in range(len(self.fc_layers)-1):
            x = self.nonlinearity(self.fc_layers[k](x))
        y = self.fc_layers[-1](x)
        return y

class CNN(nn.Module):
    def __init__(self,
                seq_len: int,
                alphabet_len: str,
                num_filters: int = 32,
                hidden_size: int = 32,
                kernel_size: int = 3,
                 ):
        super().__init__()
    
        self.cnn_model = nn.Sequential(
            nn.Conv1d(in_channels = seq_len, out_channels = num_filters, 
                    kernel_size=kernel_size,
                    padding="valid",
                    stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels = num_filters, out_channels = num_filters, 
                    kernel_size=kernel_size,
                    padding="same",
                    stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=alphabet_len, padding='same', stride=1),
            nn.ReLU()
        )
        
        # do global max pooling manually...
        
        self.model = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
#             nn.LazyLinear(hidden_size),
#             nn.ReLU(),
            nn.Dropout(0.25),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x, _ = torch.max(x, 1)
        y = self.model(x)
        return y




# # Taken from the PyTorch Transformer tutorial
# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold
import numpy as np

class NNProxyModel(RegressorMixin):
    def __init__(self, model_type, device, seq_len, alphabet_len, **kwargs):
        super().__init__(**kwargs)
        if model_type == "MLP":
            self.model_type_func = MLP
        elif model_type == "CNN":
            self.model_type_func = CNN
        else:
            raise NotImplementedError

        self.seq_len = seq_len
        self.alphabet_len = alphabet_len
        self.model = self.model_type_func(seq_len=seq_len, alphabet_len=alphabet_len)
        self.batch_size = 50
        self.nEpochs = 10
        self.device = device

        self.model.to(self.device)

    def fit(self, X, y, **kwargs):
        """
            X: (nData, seq_len * alphabet)
            y: (nData, 1) -- should be... bbut sometimes (nData,)
        """
        # Re-fits... this, so new model intialisation
        # X, y = X, y.squeeze(-1)
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)

        self.model = self.model_type_func(seq_len=self.seq_len, alphabet_len=self.alphabet_len)
        self.model.to(self.device)

        opt = optim.Adam(self.model.parameters())

        nData = X.shape[0]

        for ep_i in range(self.nEpochs):
            for t1 in range(0, nData, self.batch_size):
                t2 = min(t1 + self.batch_size, nData)

                batch_x = X[t1:t2].to(self.device)
                batch_true_y = y[t1:t2].to(self.device) # Shape mismatch later...

                batch_pred_y = self.model(batch_x)
                loss = F.mse_loss(batch_pred_y, batch_true_y)
                
                opt.zero_grad()
                loss.backward()
                opt.step()

        return self




    def predict(self, X, **kwargs): 
        """
            X: (nData, seq_len * alphabet)
            return:
                y (numpy array):  (nData, )
        """

        # X = torch.tensor(X)
        nData = X.shape[0]

        predictions = []
        for t1 in range(0, nData, self.batch_size):
            t2 = min(t1 + self.batch_size, nData)

            batch_x = X[t1:t2].to(self.device)

            batch_pred_y = self.model(batch_x)
            predictions.append(batch_pred_y)

        return torch.cat(predictions, 0).squeeze(-1).detach().cpu().numpy()

    