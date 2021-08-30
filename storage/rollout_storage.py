import torch
from storage.base import BaseStorage
import utils.helpers as utl
from torch.nn import functional as F

class RolloutStorage(BaseStorage):

    def __init__(self, 
                 num_samples,
                 state_dim,
                 action_dim, 
                 hidden_dim,
                 num_steps,
                 device
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
        self.rewards = torch.zeros(num_steps+1, num_samples, 1).to(device)
        self.log_probs = torch.zeros(num_steps+1, num_samples, 1).to(device)
        self.dones = torch.zeros(num_steps+1, num_samples, 1).to(device)
        self.masks = torch.zeros(num_steps+1, num_samples, 1).to(device) # 1 when action is performed, 0 when episode ends.


        self.returns = torch.zeros(num_steps+1, num_samples, 1).to(device)



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

        self.states[self.curr_timestep][self.curr_sample].copy_(state.clone().detach())
        self.next_states[self.curr_timestep][self.curr_sample].copy_(next_state.clone().detach())
        self.actions[self.curr_timestep][self.curr_sample].copy_(action.clone().detach())
        self.rewards[self.curr_timestep][self.curr_sample].copy_(reward.clone().detach())
        self.log_probs[self.curr_timestep][self.curr_sample].copy_(log_prob.clone()) # Leo: Used to differentiate through
        self.dones[self.curr_timestep][self.curr_sample].copy_(done.clone().detach())

        self.curr_timestep = self.curr_timestep + 1


    def after_traj(self):
        self.curr_timestep = 0
        self.curr_sample = (self.curr_sample + 1) % self.num_samples

    def compute_returns(self, gamma = 1.00):
        self.returns[self.num_steps] = self.rewards[self.num_steps]
        for step in reversed(range(self.num_steps)):
            self.returns[step] = self.rewards[step] + gamma * self.returns[step+1]

    def after_rollouts(self):

        self.masks[0] = 1
        for i in range(1, self.num_steps+1):
            self.masks[i] = self.masks[i-1] - self.dones[i-1]



    def compute_log_probs(self, policy):
        
        # Leo: Can parallelise this...
        for j in range(self.num_samples):
            hidden_state = None
            for i in range(self.num_steps+1):

            #     st = F.one_hot(self.states[i][j].long(), num_classes=21).float()[i-1]  # Leo: To check if this is correct...
            #     value, self.log_probs[i][j], dist_entropy, hidden_state = policy.evaluate_actions(st.unsqueeze(0).unsqueeze(0), hidden_state, None, self.actions[i][j].int().item())



                # value, action_log_probs, dist_entropy, rnn_hxs = policy.evaluate_actions(st.unsqueeze(0).unsqueeze(0), hidden_state, None, self.actions[i][j].int().item())


                st = self.states[i][j].flatten()
                value, action_log_probs, dist_entropy, hidden_state = policy.evaluate_actions(st.unsqueeze(0), hidden_state, self.masks[i][j].unsqueeze(0), self.actions[i][j].int().unsqueeze(0))

                self.log_probs[i][j] = action_log_probs[0]


