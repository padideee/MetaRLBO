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

    def forward(self, input):
        """
          Returns a random action from lo... hi-1
        """

        # return Independent(Normal(loc=mu, scale=scale), 1)
        return torch.randint(self.lo, self.hi)

