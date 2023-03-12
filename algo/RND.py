import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


def get_int_r(x):
    # model = MLP(input_size,
    #             hidden_size,
    #             output_size)
    # model.apply(initialize_weights)
    #
    # rand_model = MLP(input_size, hidden_size, output_size)
    # rand_model.apply(initialize_weights)
    # for param in rand_model.parameters():
    #     param.requires_grad = False
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    log_error = []

    # int_rew = torch.mean(torch.square(model(x) - rand_model(x)))
    # log_error.append(int_rew.item())
    # 
    # optimizer.zero_grad()
    # int_rew.backward()
    # optimizer.step()

    return log_error
