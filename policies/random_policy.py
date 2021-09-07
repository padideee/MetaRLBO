import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions.categorical import Categorical

from policies.policy import Policy

class RandomPolicy(Policy):
    """Random Policy
    """
    def __init__(self,
                 input_size,
                 output_size,
                 num_actions):
        super(RandomPolicy, self).__init__(input_size=input_size,
                                              output_size=output_size)


        self.lo = 0
        self.hi = num_actions

    def act(self, input, hidden_state, masks=None):


        batch_size = input.shape[0]
        return None, torch.randint(self.lo, self.hi, size = (batch_size, )), torch.log(torch.ones(size = (batch_size, )) / (self.hi - self.lo + 1)), None