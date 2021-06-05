from oracles.gridworld import GridworldImaginedOracle
from oracles.molecule import MoleculeImaginedOracle

import utils.helpers as utl
from models.online_storage import RolloutStorage
from models.query_storage import QueryStorage

from models.policy import Policy
from algo.ppo import PPO
import torch
from environments.envs import make_vec_envs
import numpy as np
from tqdm import tqdm

from acquisition_functions import UCB
from torch.utils.data import DataLoader
import time
from collections import deque

from stable_baselines3.common.running_mean_std import RunningMeanStd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetaLearner:
    """
    Meta-Learner class with the main training loop
    """

    def __init__(self, config):
        self.config = config


        D_AMP = QueryStorage(self.config["max_num_queries"]) # TODO: Replace with https://github.com/padideee/MBRL-for-AMP/blob/main/main.py

        self.policy = NormalMLPPolicy(...) # 
        self.D_train = QueryStorage(self.config["max_num_queries"])
        # self.D_query = QueryStorage(self.config["query_storage_size"])
        # self.D_meta_query = RolloutStorage(self.config["num_meta_proxy_samples"] * self.config["num_proxies"])


        self.true_oracle = TrueOracle(D_AMP)
        self.true_oracle_model = utl.get_true_oracle_model(self.config)

        self.proxy_oracles = [AMPProxyOracle(self.D_train) for j in range(self.config["num_proxies"])]
        self.proxy_oracle_models = [utl.get_proxy_oracle_model(self.config) for j in range(self.config["num_proxies"])]



    def meta_update(self):
        pass



    def run(self):

        """
            TODO:
             - Datasets Initialization
             - Sampling of molecules

             - Loss Calculation
             - Meta-update

        """

        self.true_oracle.fit(self.true_oracle_model, self.D_AMP)
        updated_params = [None for _ in range(self.config["num_proxies"])]

        for i in range(self.config["num_meta_updates"]):
            # self.D_query = ...
            self.D_meta_query = RolloutStorage(self.config["num_meta_proxy_samples"] * self.config["num_proxies"])


            # Sample molecules to train proxy oracles
            if i == 0:
                sampled_mols = ... # Sample from true env. using random policy (num_starting_mols, dim of mol)

                sampled_mols_scores = true_oracle.query(self.true_oracle_model, sampled_mols)

                # Add to storage

                self.D_train.insert(sampled_mols, sampled_mols_scores)

            else:

                for j in range(self.config["num_proxies"]):
                    sampled_mols = ... # Sample from policies -- preferably make this parallelised in the future
                    sampled_mols_scores = self.proxy_oracles[j].query(self.proxy_oracle_models[j], sampled_mols)


                    self.D_train.insert(sampled_mols, sampled_mols_scores)


            # Fit proxy oracles
            for j in range(self.config["num_proxies"]):
                self.proxy_oracles[j].fit(self.proxy_oracle_models[j])




            # Proxy(Task)-specific updates
            for j in range(self.config["num_proxies"]):


                self.D_j = RolloutStorage(self.config["num_proxy_samples"])


                sampled_mols = ... # Sample from policy[j]

                sampled_mols_scores = self.proxy_oracles[j].query(self.proxy_oracle_models[j], sampled_mols)


                self.D_j.insert(sampled_mols, sampled_mols_scores)

                loss = ... # Calculate loss
                updated_params[j] = self.policy.update_params(loss) # Tristan's update_params for MAML-RL "https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/policies/policy.py"


            for j in range(self.config["num_proxies"]):

                sampled_mols = ... # Sample from policies using (update_params)

                sampled_mols_scores = self.proxy_oracles[j].query(self.proxy_oracle_models[j], sampled_mols)

                self.D_meta_query.insert(sampled_mols, sampled_mols_scores)



            # Perform meta-update
            self.meta_update()
                






    def log(self, logs):



        # Update the self.logger
        pass
