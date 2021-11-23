import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class MLP(nn.Module): # Remember to put this to device...
    def __init__(self,
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

class MLPProxyModel(RegressorMixin):
    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.model = MLP()
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

        self.model = MLP()
        self.model.to(self.device)

        opt = optim.SGD(self.model.parameters(), lr=0.01)

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




    def predict(self, X, **kwargs): # TODO: Rewrite all
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

    