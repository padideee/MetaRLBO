import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions.categorical import Categorical

from policies.policy import Policy

import random

class DynaPPORandomPolicy(Policy):
    """ 
        DynaPPO Random Policy
    """
    def __init__(self,
                 input_size,
                 output_size,
                 num_actions,
                 max_length=50):
        super(DynaPPORandomPolicy, self).__init__(input_size=input_size,
                                              output_size=output_size)
        self.lo = 0
        self.hi = num_actions
        self.num_actions = num_actions
        self.max_length = max_length

        self.cur_idx = 0
        self.traj_len = random.randint(1, max_length)

    def act(self, input, hidden_state, masks=None, batch_size=1):
        """
            Note this only works for batch size of 1 tentatively
        """

        action = torch.randint(self.lo, self.hi - 1, size = (batch_size, ))


        if self.cur_idx + 1 == self.traj_len:
            action = torch.tensor([self.hi - 1]) # EOS
        elif action[0].item() == self.num_actions - 1 or self.cur_idx + 1 == self.traj_len:
            # Queries reset the policy...
            self.max_length = torch.randint(1, )
            self.cur_idx = 0
        else:
            self.cur_idx += 1

        return None, action, torch.log(torch.ones(size = (batch_size, )) / (self.hi - self.lo + 1)), None