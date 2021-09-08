import torch
from storage.base import BaseStorage
import utils.helpers as utl
from torch.nn import functional as F

class RolloutStorage(BaseStorage):

    def __init__(self, 
                 num_samples,
                 state_dim,
                 action_dim, 
                 num_steps,
                 device,
                 hidden_dim=None
                 ):
        self.num_samples = num_samples
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.device = device


        self.states = torch.zeros(num_steps+1, num_samples, *state_dim).to(device) # minor issue: state_dim = (num_steps, action_dim) -- storing unnecessary info.
        self.next_states = torch.zeros(num_steps+1, num_samples, *state_dim).to(device)
        self.actions = torch.zeros(num_steps+1, num_samples, action_dim).to(device)
        if hidden_dim is not None:
            self.hidden_states = torch.zeros(num_steps+1, num_samples, hidden_dim).to(device)
        self.rewards = torch.zeros(num_steps+1, num_samples).to(device)
        self.log_probs = torch.zeros(num_steps+1, num_samples).to(device)
        self.dones = torch.zeros(num_steps+1, num_samples).to(device)
        self.masks = torch.zeros(num_steps+1, num_samples).to(device) # 1 when action is performed, 0 when episode ends.


        self.returns = torch.zeros(num_steps+1, num_samples).to(device)



        self.curr_timestep = 0
        self.curr_sample = 0

    # def insert(self, ...):
    #     raise NotImplementedError()


    def insert(self, 
               state,
               next_state, 
               action, 
               reward, 
               log_prob, 
               done):

        assert self.curr_timestep < self.num_steps + 1

        batch_size = state.shape[0]

        next_sample = min(self.curr_sample + batch_size, self.num_samples)
        sample_diff = next_sample - self.curr_sample


        self.states[self.curr_timestep][self.curr_sample:next_sample].copy_(state[:sample_diff].clone().detach())
        self.next_states[self.curr_timestep][self.curr_sample:next_sample].copy_(next_state[:sample_diff].clone().detach())
        self.actions[self.curr_timestep][self.curr_sample:next_sample].copy_(action[:sample_diff].clone().detach())
        self.rewards[self.curr_timestep][self.curr_sample:next_sample].copy_(reward[:sample_diff].clone().detach())
        self.log_probs[self.curr_timestep][self.curr_sample:next_sample].copy_(log_prob[:sample_diff].clone()) # Leo: Used to differentiate through
        self.dones[self.curr_timestep][self.curr_sample:next_sample].copy_(done[:sample_diff].clone().detach())

        self.curr_timestep = self.curr_timestep + 1


    def after_traj(self, incr):
        self.curr_timestep = 0
        self.curr_sample = (self.curr_sample + incr) % self.num_samples

    def compute_returns(self, gamma = 1.00):
        self.returns[self.num_steps] = self.rewards[self.num_steps]
        for step in reversed(range(self.num_steps)):
            self.returns[step] = self.rewards[step] + gamma * self.returns[step+1]

    def after_rollouts(self):
        # Only allows one rollout (REINFORCE)
        self.masks[0] = 1
        for i in range(1, self.num_steps+1):
            diff = (self.masks[i-1] - self.dones[i-1])
            self.masks[i] = diff * (diff > 0)



    def gae(self, values, tau=1.0, gamma = 1.0):
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        
        values = values.squeeze(-1).detach()
        # values = F.pad(values * self.masks.squeeze(-1), (0, 0, 0, 1)) # masks need to be changed if it's for meta-updating..
        values = F.pad(values, (0, 0, 0, 1))

        dones = F.pad(self.dones, (0, 0, 1, 0))

        deltas = self.rewards + gamma * values[1:] * (1-dones[:-1]) - values[:-1]
        advantages = torch.zeros_like(deltas).float()
        gae = torch.zeros_like(deltas[0]).float()
        for i in reversed(range(deltas.shape[0])):
            gae = gae * gamma * tau * (1 - dones[i]) + deltas[i]
            advantages[i] = gae

        return advantages


