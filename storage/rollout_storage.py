import torch
from storage.base import BaseStorage


class RolloutStorage(BaseStorage):

    def __init__(self, 
                 num_samples,
                 state_dim,
                 action_dim, 
                 hidden_dim,
                 num_steps,
                 ):
        self.num_samples = num_samples
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_proxies = num_proxies
        self.num_steps = num_steps


        self.states = torch.zeros(num_steps+1, num_samples, state_dim)
        self.next_states = torch.zeros(num_steps+1, num_samples, state_dim)
        self.actions = torch.zeros(num_steps+1, num_samples, action_dim)
        self.hidden_states = torch.zeros(num_steps+1, num_samples, hidden_dim)
        self.rewards = torch.zeros(num_steps+1, num_samples, 1)
        self.dones = torch.zeros(num_steps+1, num_samples, 1)



        self.curr_timestep = 0
        self.curr_sample = 0

    # def insert(self, ...):
    #     raise NotImplementedError()


    def insert(self, 
               state,
               next_state, 
               action, 
               reward, 
               done):

        assert self.curr_timestep < self.num_steps + 1

        self.states[self.curr_timestep][self.curr_sample].copy_(state)
        self.next_states[self.curr_timestep][self.curr_sample].copy_(next_state)
        self.actions[self.curr_timestep][self.curr_sample].copy_(action)
        self.rewards[self.curr_timestep][self.curr_sample].copy_(reward)
        self.dones[self.curr_timestep][self.curr_sample].copy_(done)

        self.curr_timestep = self.curr_timestep + 1

    def after_rollout(self):
        self.curr_timestep = 0
        self.curr_sample = (self.curr_sample + 1) % num_samples

