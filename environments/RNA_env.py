import gym
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from data.process_data import seq_to_encoding
from algo.diversity import diversity 
import torch.nn.functional as F
import utils.helpers as utl


class RNAEnv(gym.Env):
    def __init__(self, lambd = 0.1, radius = 2, max_length = 14, vocab_size=8, query_history = None,
                 div_metric_name="hamming", div_switch="ON"):

        self.seed()
        # Actions in AMP design are the 20 amino acids
        # For non-finite horizon case: An extra action is added to
        # represent the "end of sequence" token
        self.max_length = max_length
        self._max_episode_steps = max_length
        self.num_actions = vocab_size

        self.action_space = gym.spaces.Discrete(self.num_actions) # 20 amino acids, End of Sequence Token

        # The state at time t is given by the last t tokens in the AMP sequence
        # We limit the sequence to 50 characters of AMP, and an extra character for EOS
        self.obs_shape = [max_length, self.num_actions]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=float)

        self.start_state = torch.tensor(np.zeros(self.obs_shape)) 
        self.curr_state = self.start_state  

        self.time_step = 0 
        
        self.query_history = None
        self.evaluate = {'seq': [], 'embed_seq': [], 'reward': [], 'pred_prob': []}

        self.reward_oracle = None
        self.reward_oracle_model = None
        self.proxy_oracles = []
        self.modelbased = False

        self.div_metric_name = div_metric_name

        self.div_switch = div_switch

        self.query_reward = None



    def set_oracles(self, data):
        self.reward_oracle = data["reward_oracle"]
        self.reward_oracle_model = data["reward_oracle_model"]
        self.query_history = data["query_history"]
        self.query_reward = data["query_reward_in_env"]
        self.density_penalty = data["density_penalty"]


    def update_proxy_oracles(self, oracle):
        self.proxy_oracles = oracle

    def update_opt_method(self, modelbased):
        self.modelbased = modelbased

    def step(self, action):

        # Return: (state, reward, done, info)
        # NOTE: Reward is 0 -- to be queried outside of the environment!
        # a sequence is antimicrobial towards a certain pathogen
        reward = torch.tensor(0.0)
        pred_prob = torch.tensor([0.0, 0.0])
        density_penalty = torch.tensor(0.0)
        
        self.curr_state[self.time_step] = F.one_hot(action, num_classes = self.num_actions)

        self.time_step += 1
        done = (self.time_step == self.max_length)

        # Info must be a dictionary
        info = {"action": action, "state": self.curr_state, "pred_prob": pred_prob, "queried": False, "density_penalty": density_penalty}

        return(self.curr_state, reward, done, info)

    def reset(self):
        self.curr_state = torch.tensor(np.zeros(self.obs_shape))
        self.time_step = 0
        return self.curr_state

"""
def main():
    env = AMPEnv()
    print(check_env(env))
if __name__ == "__main__":
    main()
"""



