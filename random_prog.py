import torch
import numpy as np
import pandas as pd
from torch import optim
from tqdm import tqdm
import os
import utils.helpers as utl
import utils.reinforcement_learning as rl_utl

from storage.rollout_storage import RolloutStorage
from storage.query_storage import QueryStorage
from policies.policy import Policy
from policies.gru_policy import CategoricalGRUPolicy
from policies.random_policy import RandomPolicy
from policies.DynaPPO_random_policy import DynaPPORandomPolicy


from oracles.AMP_true_oracle import AMPTrueOracle
from oracles.proxy.AMP_proxy_oracle import AMPProxyOracle

from environments.AMP_env import AMPEnv

import higher 
from utils.tb_logger import TBLogger

from evaluation import get_test_oracle
from data.process_data import seq_to_encoding
from algo.diversity import pairwise_hamming_distance
import random
import time



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Program:

    def __init__(self, config):
        self.config = config


        # initialise tensorboard logger
        self.logger = TBLogger(self.config, self.config["exp_label"])


        utl.save_config(self.config, self.logger.full_output_folder)


        # The seq and the label from library
        # seq shape: (batch, 46*21)
        # label shape: (batch) -> in binary format:{'positive': AMP, 'negative': not AMP}
        if self.config["data_source"] == 'DynaPPO':
            from data import dynappo_data
            D_AMP = dynappo_data.get_AMP_data(self.config["mode"])
        elif self.config["data_source"] == 'Custom':
            from data.process_data import get_AMP_data
            D_AMP = get_AMP_data('data/data_train.hkl')
        else:
            raise NotImplementedError
        # path to pickle 'data/data_train.pickle'
        # TODO : the data for train and test as sys argument

        self.true_oracle = AMPTrueOracle(training_storage=D_AMP)
        self.true_oracle_model = utl.get_true_oracle_model(self.config)


        # -- BEGIN ---
        # Leo: Temporary putting this here (probably want to organise this better)


        if self.config["true_oracle"]["model_name"] == "RFC":
            self.flatten_true_oracle_input = True
        else:
            self.flatten_true_oracle_input = False



        if self.config["proxy_oracle"]["model_name"] == 'RFR' or 'KNR' or 'RR':
            self.flatten_proxy_oracle_input = True
        else:
            self.flatten_proxy_oracle_input = False


        # -- END ---
        
        self.env = AMPEnv(lambd=self.config["env"]["lambda"], radius = self.config["env"]["radius"]) # The reward will not be needed in this env.

        self.D_train = QueryStorage(storage_size=self.config["query_storage_size"], state_dim = self.env.observation_space.shape)
        self.query_history = []



        self.test_oracle = get_test_oracle()
        self.iter_idx = 0


    def run(self):

        self.true_oracle_model = self.true_oracle.fit(self.true_oracle_model, flatten_input = self.flatten_true_oracle_input)
        
        # updated_params = [None for _ in range(self.config["num_proxies"])]

        
        for self.iter_idx in tqdm(range(self.config["num_meta_updates"] // self.config["num_meta_updates_per_iter"])):

            assert self.true_oracle.query_count == self.D_train.storage_filled

            if self.true_oracle.query_count > self.config["max_num_queries"]:
                # 
                break

            logs = {} 



            
            # Sample molecules to train proxy oracles
            sampled_mols = self.sample_policy(None, self.env, self.config["num_query_per_iter"] * 2) # Sample from true env. using random policy (num_starting_mols, dim of mol)


            # Do some filtering of the molecules here...
            queried_mols, logs = self.select_molecules(sampled_mols, logs)

            # Perform the querying
            if queried_mols is not None:
                # Query the scores
                queried_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, queried_mols, flatten_input = self.flatten_true_oracle_input))

                self.D_train.insert(queried_mols, queried_mols_scores)

                # Sync query_history with D_train
                self.query_history += list(self.D_train.mols[len(self.query_history):self.D_train.storage_filled])


                logs["outer_loop/queried_mols_scores/current_batch/mean"] = queried_mols_scores.mean().item()
                logs["outer_loop/queried_mols_scores/current_batch/max"] = queried_mols_scores.max().item()

                # TODO: Log diversity here... parallelise the querying (after the unique checking)
                logs["outer_loop/queried_mols/diversity"] = pairwise_hamming_distance(queried_mols) # TODO
            
            logs[f"outer_loop/sampled_mols_scores/cumulative/mean"] = self.D_train.scores[:self.D_train.storage_filled].mean().item()
            logs[f"outer_loop/sampled_mols_scores/cumulative/max"] = self.D_train.scores[:self.D_train.storage_filled].max().item() 
            logs[f"outer_loop/sampled_mols_scores/cumulative/min"] = self.D_train.scores[:self.D_train.storage_filled].min().item()


            logs["outer_loop/num_queried/unique"] = self.true_oracle.query_count



            # Logging
            if self.iter_idx % self.config["log_interval"] == 0:

                self.log(logs)


                
                utl.save_mols(mols=self.D_train.mols[:self.D_train.storage_filled].numpy(), 
                                scores=self.D_train.scores[:self.D_train.storage_filled].numpy(),
                                folder=self.logger.full_output_folder)




    def sample_policy(self, policy, env, num_samples, policy_storage = None):
        """
            Args:
             - policy 
             - env 
             - num_samples - number of molecules to return
             - policy_storage: Optional -- Used to store the rollout for training the pollicy

            Return:
             - Molecules: Tensor of size (num_samples, dim of mol)


            When policy_storage is None, the goal is to return a set of molecules to query.
            When policy_storage is not None, the goal is to return trajectories.
        """
        if self.config["policy"]["model_name"] == "RANDOM":
            # Sample actions and end when EOS token occurs
            mols = []
            for i in range(num_samples):
                rand_mol = []
                for _ in range (self.env.max_AMP_length):
                    a = random.randint(0, self.env.num_actions - 1)
                    rand_mol.append(a)
                    if a == self.env.EOS_idx:
                        break
                rand_mol += (50 - len(rand_mol))*[self.env.EOS_idx]
                rand_mol = torch.tensor(rand_mol)
                rand_mol = torch.nn.functional.one_hot(rand_mol, num_classes=self.env.num_actions)
                mols.append(rand_mol)


        elif self.config["policy"]["model_name"] == "DynaPPO_RANDOM":
            # Sample length first, then actions
            mols = []
            for i in range(num_samples):
                rand_len = random.randint(1, self.env.max_AMP_length)

                rand_mol = [random.randint(0, self.env.num_actions - 2) for i in range(rand_len)]
                rand_mol += (50 - len(rand_mol))*[self.env.EOS_idx]
                rand_mol = torch.tensor(rand_mol)
                rand_mol = torch.nn.functional.one_hot(rand_mol, num_classes=self.env.num_actions)
                mols.append(rand_mol)
        else:
            raise NotImplementedError
        

        return_mols = torch.stack(mols, 0)
        return return_mols



    def select_molecules(self, mols, logs):
        """
            Selects the molecules to query from the ones proposed by the policy trained on each of the proxy oracle


             - Let's select the ones that have the highest score according to the proxy oracles? -- Check...
        """

        if self.config["selection_criteria"]["method"] == "RANDOM":
            print("Selection random")
            return mols[:self.config["num_query_per_iter"]], logs
        # Remove duplicate molecules... in current batch
        mols = np.unique(mols, axis = 0) 

        # Remove duplicate molecules that have already been queried...
        valid_idx = []
        for i in range(mols.shape[0]):
            tuple_mol = tuple(mols[i].flatten().tolist())
            if tuple_mol not in self.D_train.mols_set:
                valid_idx.append(i)
            else:
                print("Already queried...")
        mols = mols[valid_idx]

        if mols.shape[0] > 0:
            mols = torch.tensor(mols)


            if self.iter_idx == 0: # Special case: Random Policy
                n_query = self.config["num_initial_samples"]
            else:
                n_query = min(self.config["num_query_per_iter"], mols.shape[0])
            
            if self.config["selection_criteria"]["method"] == "RANDOM" or self.iter_idx == 0:
                # In the case that the iter_idx is 0, then we can only randomly select them...
                perm = torch.randperm(mols.shape[0])
                idx = perm[:n_query]
            else:
                raise NotImplementedError

            selected_mols = mols.clone()[idx]

            return selected_mols, logs

        else:
            return None, logs






    def log(self, logs):
        """
            Tentatively Tensorboard, but perhaps we want WANDB
        """
        num_queried = (self.iter_idx + 1) * self.config["num_query_per_iter"]

        for k, v in logs.items():
            self.logger.add(k, v, num_queried)
