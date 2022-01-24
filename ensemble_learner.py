"""
    Attempts to replicate primarily the results of DynaPPO.
"""

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
from storage.seq_storage import SeqStorage
from policies.policy import Policy
from policies.gru_policy import CategoricalGRUPolicy
from policies.random_policy import RandomPolicy

from oracles.AMP_true_oracle import AMPTrueOracle
from oracles.AltIsing_true_oracle import AltIsingTrueOracle
from oracles.CLAMP_true_oracle import CLAMPTrueOracle
from oracles.proxy.AMP_proxy_oracle import AMPProxyOracle

from environments.AMP_env import AMPEnv
from environments.parallel_envs import make_vec_envs
import higher
from utils.tb_logger import TBLogger

# from evaluation import get_test_oracle
from data.process_data import seq_to_encoding
from algo.diversity import pairwise_hamming_distance
import time

from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

from utils.optimization import conjugate_gradient
from utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)
from algo.baseline import LinearFeatureBaseline
from storage.online_storage import OnlineStorage
from storage.query_storage import QueryStorage
from algo.ppo import PPO
# from algo.reinforce import REINFORCE

from utils import filtering
from algo.diversity import diversity 
from data import dynappo_data, clamp_data
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EnsembleLearner:
    """
    EnsembleLearner class with the main training loop -- Ensemble of Policies
    """

    def __init__(self, config, use_logger=True):
        self.config = config
        self.device = device
        torch.multiprocessing.set_start_method('spawn')
        # initialise tensorboard logger
        self.use_logger = use_logger
        if self.use_logger: 
            self.logger = TBLogger(self.config, self.config["exp_label"]) if self.use_logger else None
            utl.save_config(self.config, self.logger.full_output_folder)
        else:
            self.logger = None


        self.config = utl.setup_task_configs(self.config) # Setup for Ising20/50 configs.

        if self.config["task"] == "AMP-v0":
            if self.config["data_source"] == 'DynaPPO':
                D_AMP = dynappo_data.get_AMP_data(self.config["mode"])
            elif self.config["data_source"] == 'Custom':
                from data.process_data import get_AMP_data
                D_AMP = get_AMP_data('data/data_train.hkl')
            else:
                raise NotImplementedError

            self.true_oracle = AMPTrueOracle(training_storage=D_AMP)
            if self.config["mode"] == "test":
                fname = "data/metarlbo_rfc_not-alibicani.pkl"
            elif self.config["mode"] == "val":
                fname = "data/metarlbo_rfc_alibicani.pkl"
            with open(fname, "rb") as f:
                self.true_oracle_model = pickle.load(f)

            # self.true_oracle_model = utl.get_true_oracle_model(self.config)
        elif self.config["task"] == "CLAMP-v0":

            if self.config["mode"] == "val" and self.config["CLAMP"]["use_pretrained_model"]:
                from common_evaluation.clamp_common_eval.defaults import get_test_oracle
                self.true_oracle = CLAMPTrueOracle(self.config["CLAMP"]["true_oracle_model"])
                self.true_oracle_model = get_test_oracle(source=self.config["CLAMP"]["data_source"], 
                                                        model=self.config["CLAMP"]["true_oracle_model"], 
                                                        feature="AlBert",
                                                        device=device) #TODO: Set this up to either be RandomForest... or MLP
            elif self.config["mode"] == "val" and not self.config["CLAMP"]["use_pretrained_model"]:
                D_AMP = clamp_data.get_CLAMP_data(self.config["mode"])
                self.true_oracle = AMPTrueOracle(training_storage=D_AMP) # Use the RFC classifier from AMP task
                self.true_oracle_model = utl.get_true_oracle_model(self.config)
            elif self.config["mode"] == "test":
                # Use Moksh + Emmanuels oracle!
                if self.config["CLAMP"]["true_oracle_model"] == "GFN":
                    # Use Moksh + Emmanuels oracle! 
                    from oracles.gflownet_oracle import get_proxy as get_oracle # Use their proxy as our oracle
                    self.true_oracle = CLAMPTrueOracle(self.config["CLAMP"]["true_oracle_model"])
                    print("Using GFlownet Oracle... First performing training")
                    self.true_oracle_model = get_oracle()
                else:
                    D_AMP = clamp_data.get_CLAMP_data(self.config["mode"])
                    self.true_oracle = AMPTrueOracle(training_storage=D_AMP) # Use the RFC classifier from AMP task
                    self.true_oracle_model = utl.get_true_oracle_model(self.config)


            else:
                raise NotImplementedError
        elif 'AltIsing' in self.config["task"]:

            self.true_oracle = AltIsingTrueOracle()
            self.true_oracle_model = utl.get_true_oracle_model(self.config)
        elif self.config['task'] == 'RNA14-v0':
            from oracles.RNA_true_oracle import RNATrueOracle
            self.true_oracle = RNATrueOracle()
            self.true_oracle_model = utl.get_true_oracle_model(self.config)

        else:
            raise NotImplementedError

        # -- BEGIN ---
        # Leo: Temporary putting this here (probably want to organise this better)

        if self.config["true_oracle"]["model_name"] == "RFC":
            self.flatten_true_oracle_input = True
        else:
            self.flatten_true_oracle_input = False

        if self.config["proxy_oracle"]["model_name"] == "CNN":
            self.flatten_proxy_oracle_input = False
        else:    
            self.flatten_proxy_oracle_input = True

        # --- END ---

        self.env = make_vec_envs(self.config["task"],
                                             num_processes=self.config["num_processes"],
                                             seed = self.config["seed"],
                                             # End of default parameters
                                            # reward_oracle = self.true_oracle, 
                                            # reward_oracle_model = self.true_oracle_model, 
                                            lambd=self.config["env"]["lambda"],
                                          radius=self.config["env"]["radius"],
                                          div_metric_name=self.config["diversity"]["div_metric_name"],
                                          div_switch=self.config["diversity"][
                                              "div_switch"])
                            

        # Molecules that have been queried w/ their scores
        self.D_train = QueryStorage(storage_size=self.config["query_storage_size"],
                                    state_dim=self.env.observation_space.shape)
        self.D_train.mols_set.add("") # Add the empty mol...
        self.sampled_dataset = SeqStorage()

        # Molecules that have been queried (used for diversity)
        self.query_history = []
        self.proxy_oracles = [] # Leo: TODO -- We need to create an "ensemble" proxy oracle class!
        self.proxy_oracle_models = []


        self.reset_policy()


        
        self.iter_idx = 0
        self.best_batch_mean = 0
        self.best_batch_max = 0


        self.total_time_sampling = 0

    def reset_policy(self):
        print("New Round: Resetting POLICY")
        self.policies = []
        self.agents = []
        for _ in range(self.config["num_policies"]):
            if self.config["policy"]["model_name"] == "GRU":
                policy = CategoricalGRUPolicy(num_actions=self.env.action_space.n + 1,
                                                   hidden_size=self.config["policy"]["model_config"]["hidden_dim"],
                                                   state_dim=self.env.action_space.n + 1,
                                                   state_embed_dim=self.config["policy"]["model_config"][
                                                       "state_embedding_size"],
                                                   ).to(device)
            elif self.config["policy"]["model_name"] == "MLP":
                from policies.mlp_policy import Policy as MLPPolicy
                policy = MLPPolicy((self.env.observation_space.shape[0] * self.env.observation_space.shape[1],),
                                        self.env.action_space).to(device)
            elif self.config["policy"]["model_name"] == "RANDOM":
                policy = RandomPolicy(input_size=self.env.observation_space.shape, output_size=1,
                                           num_actions=self.env.action_space.n).to(device)
            else:
                raise NotImplementedError
            self.policies.append(policy)

            ppo_config = self.config["ppo_config"]


            if self.config["optimizer"] == "PPO":

                agent = PPO(  # TODO: Change to non recurrent
                    policy,
                    ppo_config["clip_param"],
                    ppo_config["ppo_epoch"],
                    ppo_config["num_mini_batch"],
                    value_loss_coef=ppo_config["value_loss_coef"],
                    entropy_coef=ppo_config["entropy_coef"],
                    lr=ppo_config["lr"],
                    eps=ppo_config["eps"],
                    max_grad_norm=ppo_config["max_grad_norm"])
            elif self.config["optimizer"] == "REINFORCE":
                agent = REINFORCE(...) # TODO
            else:
                raise NotImplementedError

            self.agents.append(agent)

        self.policy_storages = [
            OnlineStorage(
                self.config["policy"]["num_steps"],
                self.config["num_processes"],
                self.env.observation_space.shape,
                self.env.action_space, # Action is a single discrete variable!
                self.policies[0].recurrent_hidden_state_size)  # TODO: Change to non recurrent
            for _ in range(self.config["num_policies"])
            ]
        for storage in self.policy_storages:
            storage.to(device)





    def run(self):
        start_time = time.time()
        self.true_oracle_model = self.true_oracle.fit(self.true_oracle_model,
                                                      flatten_input=self.flatten_true_oracle_input)

        # updated_params = [None for _ in range(self.config["num_proxies"])]



        for self.iter_idx in tqdm(range(self.config["num_meta_updates"] // self.config["num_updates_per_iter"])):

            # assert self.true_oracle.query_count == self.D_train.storage_filled

            if self.true_oracle.query_count > self.config["max_num_queries"]:
                #
                break

            logs = {}

            # Sample molecules to train proxy oracles
            if self.iter_idx == 0:

                random_policy = RandomPolicy(input_size=self.env.observation_space.shape, output_size=1,
                                             num_actions=self.env.action_space.n).to(device)
                sampled_mols = self.sample_policy(random_policy, oracle=self.true_oracle, oracle_model=self.true_oracle_model, num_samples = self.config[
                    "num_initial_samples"] * 2, num_steps=self.config["policy"]["num_steps"], density_penalty = False)  # Sample from true env. using random policy (num_starting_mols, dim of mol)

            else:


                # Proxy -- used for training
                self.proxy_oracles = [AMPProxyOracle(training_storage=self.D_train, p=self.config["proxy_oracle"]["p"])  for _ in range(self.config["num_policies"])]
                self.proxy_oracle_models = [utl.get_proxy_oracle_model(self.config) for _ in range(self.config["num_policies"])]

                time_st = time.time()
                for j in range(self.config["num_policies"]):
                    self.proxy_oracle_models[j] = self.proxy_oracles[j].fit(self.proxy_oracle_models[j], flatten_input=self.flatten_proxy_oracle_input)
                logs["timing/fitting_proxy_oracle"] = time.time() - time_st


                st_time = time.time()
                for policy, policy_storage, agent, proxy_oracle, proxy_oracle_model in zip(self.policies, self.policy_storages, self.agents, self.proxy_oracles, self.proxy_oracle_models):
                    self.train_policy(logs, policy, policy_storage, agent, proxy_oracle, proxy_oracle_model)

                logs["timing/updates"] = time.time() - st_time
                
                st_time = time.time()

                # Sample from generator
                sampled_mols, logs, sampled_info = self.sample_query_mols(logs, self.config["num_query_proxies"], self.config["num_samples_per_proxy"])

                logs["timing/sample_query_mols"] = time.time() - st_time
                st_time = time.time()

                sampled_mols = torch.cat(sampled_mols, dim=0)

                sampled_mols_strings = utl.mol_to_string_encoding(self.config, sampled_mols)
                self.sampled_dataset.insert(sampled_mols_strings, sampled_info['query_proxy_idx'], sampled_info['query_round'])

            st_time = time.time()

            # Selection of sequences
            if self.iter_idx == 0:
                queried_mols, logs = self.select_molecules(sampled_mols, logs, self.config["num_initial_samples"])
            else:
                queried_mols, logs = self.select_molecules(sampled_mols, logs, self.config["num_query_per_iter"])
            logs["timing/selecting_molecules"] = time.time() - st_time
            logs["timing/total_time_sampling"] = self.total_time_sampling

            # Query the sequences and store in queried sequence dataset
            if queried_mols is not None:
                queried_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, queried_mols,
                                                                          flatten_input=self.flatten_true_oracle_input))
                
                queried_mols_strings = utl.mol_to_string_encoding(self.config, queried_mols)
                self.D_train.insert(queried_mols, queried_mols_scores, queried_mols_strings, query_round=self.iter_idx)

                # Sync query_history with D_train
                self.query_history += list(self.D_train.mols[len(self.query_history):self.D_train.storage_filled])

                batch_mean, batch_max = queried_mols_scores.mean().item(), queried_mols_scores.max().item()

                logs["outer_loop/queried_mols_scores/current_batch/mean"] = batch_mean
                logs["outer_loop/queried_mols_scores/current_batch/max"] = batch_max

                # TODO: Log diversity here... parallelise the querying (after the unique checking)
                logs["outer_loop/queried_mols/diversity"] = pairwise_hamming_distance(queried_mols)

            cumul_min, cumul_mean, cumul_max = self.D_train.scores[:self.D_train.storage_filled].min().item(), self.D_train.scores[:self.D_train.storage_filled].mean().item(), self.D_train.scores[:self.D_train.storage_filled].max().item()

            logs[f"outer_loop/sampled_mols_scores/cumulative/mean"] = cumul_mean
            logs[f"outer_loop/sampled_mols_scores/cumulative/max"] = cumul_max
            logs[f"outer_loop/sampled_mols_scores/cumulative/min"] = cumul_min

            logs["outer_loop/num_queried/unique"] = self.true_oracle.query_count

            # Logging
            if self.iter_idx % self.config["log_interval"] == 0:

                # if self.best_batch_mean < batch_mean:
                #     self.best_batch_mean = batch_mean
                #     save_path = os.path.join(self.logger.full_output_folder, f"best_batch_mean_policy.pt")
                #     torch.save(self.policy.state_dict(), save_path)

                # if self.best_batch_max < batch_max:
                #     self.best_batch_max = batch_max
                #     save_path = os.path.join(self.logger.full_output_folder, f"best_batch_max_policy.pt")
                #     torch.save(self.policy.state_dict(), save_path)


                logs["timing/time_running"] = time.time() - start_time
                self.print_timing(logs)
                self.log(logs)

                utl.save_queried_mols(dataset = self.D_train, folder=self.logger.full_output_folder)
                utl.save_sampled_mols(dataset = self.sampled_dataset, folder=self.logger.full_output_folder)

            if (self.iter_idx + 1) % self.config["save_interval"] == 0:
                print(f"Saving model at iter-{self.iter_idx}...")
                for i in range(self.config["num_policies"]):
                    save_path = os.path.join(self.logger.full_output_folder, f"policy_{self.iter_idx}_ens-{i}.pt")
                    torch.save(self.policies[i].state_dict(), save_path)


    def train_policy(self, logs, policy, policy_storage, agent, proxy_oracle, proxy_oracle_model):

        info = {
                'query_proxy_idx': [],
                'query_round': [],
                }

        for _ in tqdm(range(self.config["num_updates_per_iter"])): #TODO: Set this in config... num_updates...
            # episodes = RolloutStorage(num_processes=self.config["num_processes"],
            #                                    state_dim=self.env.observation_space.shape,
            #                                    action_dim=1,  # Discrete value
            #                                    num_steps=self.config["policy"]["num_meta_steps"],
            #                                    device=device
            #                                    )
            self.sample_policy(policy, proxy_oracle, proxy_oracle_model, num_steps=self.config["policy"]["num_meta_steps"],
                               policy_storage=policy_storage, density_penalty = self.config["outerloop"]["density_penalty"])
            
            # Update the policy
            value_loss, action_loss, dist_entropy = agent.update(policy_storage)


            policy_storage.after_update()
        logs["general/value_loss"] = value_loss
        logs["general/action_loss"] = action_loss
        logs["general/dist_entropy"] = dist_entropy

        return logs

    def sample_query_mols(self, logs, num_policies, num_samples):

        sampled_mols = []
        info = {
                'query_proxy_idx': [],
                'query_round': [],
                }

        for j, (policy, proxy_oracle, proxy_oracle_model) in enumerate(zip(self.policies, self.proxy_oracles, self.proxy_oracle_models)):
            # Sample mols (and query) for training the proxy oracles later
            generated_mols = self.sample_policy(policy, proxy_oracle, proxy_oracle_model, 
                                                    num_steps=self.config["policy"]["num_steps"],
                                                    num_samples=num_samples).detach()
            sampled_mols.append(generated_mols)  # Sample from policies -- preferably make this parallelised in the future
            info['query_proxy_idx'] = info['query_proxy_idx'] + [j] * len(generated_mols)

        info['query_round'] = [self.iter_idx] * len(info['query_proxy_idx'])
        return sampled_mols, logs, info 

    def sample_policy(self, policy, oracle, oracle_model, num_steps, num_samples=None, policy_storage=None, density_penalty=False):
        """
            Args:
             - policy
             - oracle
             - oracle_model
             - num_samples - number of molecules to return
             - policy_storage: Optional -- Used to store the rollout for training the pollicy
            Return:
             - Molecules: Tensor of size (num_samples, dim of mol)
            When policy_storage is None, the goal is to return a set of molecules to query.
            When policy_storage is not None, the goal is to store trajectories in policy_storage.
        """

        time_st = time.time()

        state_dim = self.env.observation_space.shape
        if num_samples is not None:
            return_mols = torch.zeros(num_samples, *state_dim)
        else:
            return_mols = None
        query_reward = policy_storage is not None
        curr_sample = 0
        # data = {"reward_oracle": oracle,
        #         "reward_oracle_model": oracle_model,
        #         "query_history": self.query_history,
        #         "query_reward_in_env": query_reward and self.config["query_reward_in_env"], 
        #         "density_penalty": density_penalty}
        # self.env.set_oracles(data)

        break_loop = False
        while not break_loop:
            """
                Loop if needed to generate more molecules to query...
            """


            state = torch.tensor(self.env.reset()).float()  # Returns (num_processes, 51, 21) 
            hidden_state = torch.zeros((state.shape[0], 1))
            for stepi in range(num_steps):

                masks = None
                st = state
                st = seq_to_encoding(st).flatten(-2, -1).to(device)  
                value, action, log_prob, hidden_state = policy.act(st, hidden_state, masks=masks)
                action = action.detach().cpu()

                next_state, reward, done, infos = self.env.step(action) 


                done = torch.tensor(done).float()
                next_state = torch.tensor(next_state).float()
                reward = torch.tensor(reward)

                if policy_storage is not None:
                    # If done then clean the history of observations.
                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done]) # TODO
                    bad_masks = torch.FloatTensor(
                        [[0.0] if 'bad_transition' in info.keys() else [1.0]
                         for info in infos])

                    done = torch.FloatTensor(done).unsqueeze(-1)
                    # print(reward, reward.shape)

                    policy_storage.insert(obs=seq_to_encoding(state).flatten(-2, -1), raw_obs=state, recurrent_hidden_states=hidden_state, actions=action,
                                               action_log_probs=log_prob, value_preds=value, rewards=reward.unsqueeze(-1), masks=masks, bad_masks=bad_masks, dones=done, next_obs=seq_to_encoding(next_state).flatten(-2, -1), raw_next_obs=next_state)
                    # policy_storage.insert(state=state.detach().clone(),
                    #                       next_state=next_state.detach().clone(),
                    #                       action=action.detach().clone(),
                    #                       reward=reward.detach().clone(),
                    #                       done=done.detach().clone())

                # reset environments that are done
                done_indices = np.argwhere(done.cpu().flatten()).flatten()
                if len(done_indices) > 0:

                    if policy_storage is None:
                        #Save the molecule as a "return_mol"
                        for i in done_indices:
                            return_mols[curr_sample] = next_state[i].clone()
                            curr_sample += 1
                            if curr_sample == num_samples:
                                self.total_time_sampling += time.time() - time_st
                                return return_mols # Return if sufficient number of molecules are found

                    next_state = utl.reset_env(self.env, self.config, done_indices, next_state)

                state = next_state.clone()

            if policy_storage is not None:
                break_loop = True

        if query_reward and not self.config["query_reward_in_env"]:
            bool_idx = policy_storage.dones[:-1].bool().squeeze(-1)
            query_states = policy_storage.raw_next_obs[:-1][bool_idx] # Leo: TODO

            dens = diversity(query_states.cpu(), 
                                self.query_history, 
                                div_switch=self.config["diversity"]["div_switch"], 
                                radius=self.config["env"]["radius"], 
                                div_metric_name=self.config["diversity"]["div_metric_name"]).get_density()            

            query_scores = oracle.query(oracle_model, query_states.cpu(), flatten_input=self.flatten_proxy_oracle_input)

            policy_storage.rewards[bool_idx] = (torch.tensor(query_scores).float().to(device) - self.config["env"]["lambda"] * dens.to(device)).unsqueeze(-1) # TODO: Set the rewards to include the density penalties...
            

        if policy_storage is not None:
            next_value = policy.get_value(
               policy_storage.obs[-1], policy_storage.recurrent_hidden_states[-1],
               policy_storage.masks[-1]).detach()

            policy_storage.compute_returns(next_value, self.config["ppo_config"]["use_gae"],
                                               self.config["ppo_config"]["gamma"],
                                               self.config["ppo_config"]["gae_lambda"],
                                               self.config["ppo_config"]["use_proper_time_limits"])
            # # TODO: If the meta optimiser doesn't use bootstrapping... then we do this
            # policy_storage.after_rollouts() # TODO: Does this need to be activated for the results?

        self.total_time_sampling += time.time() - time_st

        return return_mols

    def select_molecules(self, mols, logs, n_query, use_diversity_metric=True):
        """
            Selects the molecules to query from the ones proposed by the policy trained on each of the proxy oracle
             - Let's select the ones that have the highest score according to the proxy oracles? -- Check...
        """


        return mols[:min(n_query, mols.shape[0])], logs # TODO: Currently hardcode return molecules!

        # from data.dynappo_data import enc_to_seq
        # # Remove duplicate molecules... in current batch
        # mols = np.unique(mols, axis=0)

        # # Remove duplicate molecules that have already been queried...
        # valid_idx = []
        # for i in range(mols.shape[0]):
        #     seq = enc_to_seq(torch.tensor(mols[i]))
        #     if seq.find(">") != -1:
        #         seq = seq[:seq.find(">")]

        #     if seq not in self.D_train.mols_set:
        #         valid_idx.append(i)
        #     else:
        #         print("Already queried...")

        # mols = mols[valid_idx] # Filter out the duplicates

        # if mols.shape[0] > 0:
        #     mols = torch.tensor(mols)

        #     n_query = min(n_query, mols.shape[0])

        #     logs, scores = filtering.get_scores(self.config, mols, self.proxy_oracles, self.proxy_oracle_models, self.flatten_proxy_oracle_input, logs, iter_idx = self.iter_idx)

        #     _, sorted_idx = torch.sort(scores, descending = True)

        #     sorted_mols = mols.clone()[sorted_idx]
        #     selected_mols = filtering.select(self.config, sorted_mols, n_query, use_diversity_metric=use_diversity_metric)

        #     return selected_mols, logs
        # else:
        #     return None, logs








    def log(self, logs):
        """
            Tentatively Tensorboard, but perhaps we want WANDB
        """
        num_queried = self.config["num_initial_samples"] + self.iter_idx * self.config["num_query_per_iter"]

        # log the average weights and gradients of all models (where applicable)
        for [model, name] in [
            [self.policies[0], 'policy']
        ]:
            if model is not None:
                param_list = list(model.parameters())
                param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
                self.logger.add('weights/{}/mean'.format(name), param_mean, num_queried)

                if name == 'policy':
                    self.logger.add('weights/policy/std', param_list[0].data.mean(), num_queried)

                if param_list[0].grad is not None:
                    param_list_grad = [param_list[i].grad for i in range(len(param_list))]
                    param_list_grad = list(filter(lambda x: x is not None, param_list_grad))

                    param_grad_mean = np.mean(
                        [param_list_grad[i].cpu().numpy().mean() for i in range(len(param_list_grad))])

                    self.logger.add('gradients/{}'.format(name), param_grad_mean, num_queried)

        for k, v in logs.items():
            self.logger.add(k, v, num_queried)

    def print_timing(self, logs):
        for k, v in logs.items():
            if "timing" in k:
                print(v, k)
