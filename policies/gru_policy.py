import torch.nn as nn
from torch.nn import functional as F
import utils.helpers as utl
import numpy as np
import torch


class CategoricalGRUPolicy(nn.Module):
    def __init__(self,
                num_actions,
                hidden_size,
                state_dim,
                state_embed_dim,
                layers_after_gru = (),
                ):
        super(CategoricalGRUPolicy, self).__init__()

        self.num_actions = num_actions

        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)

        curr_input_size = state_embed_dim
        self.gru = nn.GRU(input_size=curr_input_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          )

        # fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]

        # Prediction

        self.fc_action = nn.Linear(curr_input_dim, num_actions)



        self.softmax = nn.Softmax(dim=-1)


    def forward(self, state, hidden_state=None):
        curr_input = state

        curr_input = self.state_encoder(curr_input)

        if hidden_state is not None:
            curr_input, gru_h = self.gru(curr_input, hidden_state)
        else:
            curr_input, gru_h = self.gru(curr_input)


        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            curr_input = F.relu(self.fc_after_gru[i](curr_input))



        act_prob = self.softmax(self.fc_action(curr_input))

        return act_prob, gru_h

    def act(self, state, hidden_state=None):
        act_prob, gru_h = self.forward(state, hidden_state)


        # Tentatively hardcoded w/ batch of 1

        np_act_prob = act_prob.detach()[0][0].numpy()
        # Renormalize
        np_act_prob /= np_act_prob.sum()

        a = np.random.choice(self.num_actions, p=np_act_prob)
        return a, torch.log(act_prob[0][0][a]), gru_h


    def evaluate_actions(self, state, hidden_state, masks, action):
        
        act_prob, gru_h = self.forward(state, hidden_state)


        # Placeholders
        value = 0
        dist_entropy = 0

        return value, torch.log(act_prob[0][0][action]), dist_entropy, gru_h


