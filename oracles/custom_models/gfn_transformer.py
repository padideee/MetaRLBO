import math
import torch
import torch.nn as nn



class GFNTransformer(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid,
                 num_layers, num_head, max_len=60, dropout=0.1):
        super().__init__()
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 1)
        self.embedding = nn.Embedding(num_tokens, num_hid)
        encoder_layers = nn.TransformerEncoderLayer(num_hid, num_head, num_hid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = nn.Linear(num_hid, num_outputs)

    def forward(self, x, mask):
        u = x
        x = self.embedding(x)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        pooled_x = x[0, :] # This is weird... but this is what BERT calls pooling?
        # indeed doing e.g. this:
        #_mask = (1-mask.float())
        #pooled_x = (x * _mask.T.unsqueeze(2)).sum(0) / _mask.sum(1).unsqueeze(1)
        # seems to no be as good? (Well, the max reward is lower but loss is similar..)
        y = self.output(pooled_x)
        return y



# Taken from the PyTorch Transformer tutorial
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)