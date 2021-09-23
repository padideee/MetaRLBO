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

from oracles.AMP_true_oracle import AMPTrueOracle
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

from utils import filtering
from algo.diversity import diversity 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetaLearner:
    """
    Meta-Learner class with the main training loop
    """

    def __init__(self, config):
        self.config = config

        # initialise tensorboard logger
        self.logger = TBLogger(self.config, self.config["exp_label"])

        utl.save_config(self.config, self.logger.full_output_folder)

        # The seq and the label from library
        # seq shape: (batch, 46*21)
        # label shape: (batch) -> in binary format:{'positive': AMP, 'negative': not AMP}


        if self.config["task"] == "AMP-v0":
            if self.config["data_source"] == 'DynaPPO':
                from data import dynappo_data
                D_AMP = dynappo_data.get_AMP_data(self.config["mode"])
            elif self.config["data_source"] == 'Custom':
                from data.process_data import get_AMP_data
                D_AMP = get_AMP_data('data/data_train.hkl')
            else:
                raise NotImplementedError

            self.true_oracle = AMPTrueOracle(training_storage=D_AMP)
            self.true_oracle_model = utl.get_true_oracle_model(self.config)
        elif self.config["task"] == "CLAMP-v0":
            from common_evaluation.clamp_common_eval.defaults import get_test_oracle
            self.true_oracle = CLAMPTrueOracle()
            self.true_oracle_model = get_test_oracle(source=self.config["CLAMP"]["data_source"], 
                                                    model=self.config["CLAMP"]["true_oracle_model"], 
                                                    feature="AlBert") #TODO: Set this up to either be RandomForest... or MLP
        else:
            raise NotImplementedError


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

        # Molecules that have been queried (used for diversity)
        self.query_history = []
        self.proxy_oracles = None
        self.proxy_oracle_models = None 

        if self.config["policy"]["model_name"] == "GRU":
            self.policy = CategoricalGRUPolicy(num_actions=self.env.action_space.n + 1,
                                               hidden_size=self.config["policy"]["model_config"]["hidden_dim"],
                                               state_dim=self.env.action_space.n + 1,
                                               state_embed_dim=self.config["policy"]["model_config"][
                                                   "state_embedding_size"],
                                               ).to(device)
        elif self.config["policy"]["model_name"] == "MLP":
            from policies.mlp_policy import Policy as MLPPolicy
            self.policy = MLPPolicy((self.env.observation_space.shape[0] * self.env.observation_space.shape[1],),
                                    self.env.action_space).to(device)
        elif self.config["policy"]["model_name"] == "RANDOM":
            self.policy = RandomPolicy(input_size=self.env.observation_space.shape, output_size=1,
                                       num_actions=self.env.action_space.n).to(device)
        else:
            raise NotImplementedError

        self.baseline = LinearFeatureBaseline(input_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]).to(device) # TODO: FIX

        self.meta_opt = optim.SGD(self.policy.parameters(), lr=self.config["outer_lr"])
        
        self.iter_idx = 0
        self.best_batch_mean = 0
        self.best_batch_max = 0


        self.total_time_sampling = 0


    def run(self):
        start_time = time.time()
        self.true_oracle_model = self.true_oracle.fit(self.true_oracle_model,
                                                      flatten_input=self.flatten_true_oracle_input)

        # updated_params = [None for _ in range(self.config["num_proxies"])]



        for self.iter_idx in tqdm(range(self.config["num_meta_updates"] // self.config["num_meta_updates_per_iter"])):

            assert self.true_oracle.query_count == self.D_train.storage_filled

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

                st_time = time.time()
                # Training
                for mi in range(self.config["num_meta_updates_per_iter"]):
                    logs = self.meta_update(logs, self.config["num_proxies"])

                logs["timing/meta_updates"] = time.time() - st_time
                
                st_time = time.time()

                sampled_mols, logs = self.sample_query_mols(logs, self.config["num_query_proxies"], self.config["num_samples_per_proxy"])

                logs["timing/sample_query_mols"] = time.time() - st_time
                st_time = time.time()


                sampled_mols = torch.cat(sampled_mols, dim=0)

            st_time = time.time()

            # Do some filtering of the molecules here...
            queried_mols, logs = self.select_molecules(sampled_mols, logs)
            logs["timing/selecting_molecules"] = time.time() - st_time
            logs["timing/total_time_sampling"] = self.total_time_sampling

            # Perform the querying
            if queried_mols is not None:
                # Query the scores
                queried_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, queried_mols,
                                                                          flatten_input=self.flatten_true_oracle_input))

                self.D_train.insert(queried_mols, queried_mols_scores)

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
                # # TODO: Record the results according to the final test oracle of the new samples and the cumulative samples

                # # Leo: What do we even want to record for the test oracle?
                # #      Perhaps the likeliness of our molecule being an AMP according to that oracle?

                # # cur_batch_test_score = self.test_oracle.give_score(queried_mols, queried_mols_scores)
                # # cum_batch_test_score = self.test_oracle.give_score(self.D_train.mols[:self.D_train.storage_filled], self.D_train.scores[:self.D_train.storage_filled])

                # if queried_mols is not None:
                #     batch_test_prob = self.test_oracle.get_prob(queried_mols)

                #     logs["test_oracle/scores/current_batch/min"] = batch_test_prob.min().item()
                #     logs["test_oracle/scores/current_batch/mean"] = batch_test_prob.mean().item()
                #     logs["test_oracle/scores/current_batch/max"] = batch_test_prob.max().item()

                #     logs["test_oracle/num_mols_queried"] = queried_mols.shape[0]
                # else:
                #     logs["test_oracle/num_mols_queried"] = queried_mols.shape[0]

                # cumul_test_prob = self.test_oracle.get_prob(self.D_train.mols[:self.D_train.storage_filled])

                # logs["test_oracle/scores/cumulative/min"] = cumul_test_prob.min().item()
                # logs["test_oracle/scores/cumulative/mean"] = cumul_test_prob.mean().item()
                # logs["test_oracle/scores/cumulative/max"] = cumul_test_prob.max().item()

                # # # TODO adding to log
                # # print('Iteration {}, test oracle accuracy: {}'.format(self.iter_idx, score))

                if self.best_batch_mean < batch_mean:
                    self.best_batch_mean = batch_mean
                    save_path = os.path.join(self.logger.full_output_folder, f"best_batch_mean_policy.pt")
                    torch.save(self.policy.state_dict(), save_path)

                if self.best_batch_max < batch_max:
                    self.best_batch_max = batch_max
                    save_path = os.path.join(self.logger.full_output_folder, f"best_batch_max_policy.pt")
                    torch.save(self.policy.state_dict(), save_path)


                logs["timing/time_running"] = time.time() - start_time
                self.print_timing(logs)
                self.log(logs)

                utl.save_mols(mols=self.D_train.mols[:self.D_train.storage_filled].numpy(),
                              scores=self.D_train.scores[:self.D_train.storage_filled].numpy(),
                              folder=self.logger.full_output_folder)

            if (self.iter_idx + 1) % self.config["save_interval"] == 0:
                print(f"Saving model at iter-{self.iter_idx}...")
                save_path = os.path.join(self.logger.full_output_folder, f"policy_{self.iter_idx}.pt")
                torch.save(self.policy.state_dict(), save_path)

    def inner_loss(self, episodes, policy, params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE (possibly with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3])).
        """
        # values = self.baseline(episodes)
        # advantages = episodes.gae(values, tau=self.tau)
        # advantages = weighted_normalize(advantages, weights=episodes.mask)

        # pi = self.policy(episodes.observations, params=params)
        # log_probs = pi.log_prob(episodes.actions)
        # if log_probs.dim() > 2:
        #     log_probs = torch.sum(log_probs, dim=2)

        # loss = -weighted_mean(log_probs * advantages, dim=0, weights=episodes.mask)


        # Old Version:

        # for j in range(episodes.num_processes): # Leo: TODO -- High Priority
        #     hidden_state = None
        #     for i in range(episodes.num_steps+1):
        #         st = episodes.states[i][j]
        #         st = seq_to_encoding(st.unsqueeze(0))
        #         value, action_log_probs, dist_entropy, hidden_state = policy.evaluate_actions(st.flatten().unsqueeze(0), hidden_state, episodes.masks[i][j].unsqueeze(0), episodes.actions[i][j].int().unsqueeze(0))
        #         episodes.log_probs[i][j] = action_log_probs[0]

        # New Version (Parallelised):
        hidden_state, policy_masks = None, None
        st = seq_to_encoding(episodes.states.flatten(0, 1)).to(device)
        value, action_log_probs, dist_entropy, hidden_state = policy.evaluate_actions(st.flatten(-2, -1), hidden_state, policy_masks, episodes.actions.flatten().int())
        episodes.log_probs = action_log_probs.reshape(episodes.log_probs.shape)


        # st = seq_to_encoding(episodes.states.flatten(0, 1)).view(episodes.states.shape).to(device)  




        loss = rl_utl.reinforce_loss(episodes) + self.config[
            "entropy_reg_coeff"] * rl_utl.entropy_bonus(episodes)

        return loss

    def meta_update(self, logs, num_proxies):

        # Proxy -- used for training
        self.proxy_oracles = [AMPProxyOracle(training_storage=self.D_train, p=self.config["proxy_oracle"]["p"]) for j in
                              range(num_proxies)]
        self.proxy_oracle_models = [utl.get_proxy_oracle_model(self.config) for j in range(num_proxies)]


        self.meta_opt.zero_grad()
        inner_opt = optim.SGD(self.policy.parameters(), lr=self.config["inner_lr"])


        time_st = time.time()
        # Fit proxy oracles
        for j in range(self.config["num_proxies"]):
            self.proxy_oracle_models[j] = self.proxy_oracles[j].fit(self.proxy_oracle_models[j],
                                                                    flatten_input=self.flatten_proxy_oracle_input)
        logs["timing/fitting_proxy_oracle"] = time.time() - time_st

        episodes = []
        train_losses_s = []


        # Proxy(Task)-specific updates -- Samples the trajectories...
        for j in range(self.config["num_proxies"]):
            with higher.innerloop_ctx(
                    self.policy, inner_opt, copy_initial_weights=False
            ) as (inner_policy, diffopt):

                D_meta_query = RolloutStorage(num_processes=self.config["num_processes"],
                                                   state_dim=self.env.observation_space.shape,
                                                   action_dim=1,  # Discrete value
                                                   num_steps=self.config["policy"]["num_meta_steps"],
                                                   device=device
                                                   )


                train_episodes = []
                train_losses = []
                time_st = time.time()
                for k in range(self.config["num_inner_updates"]):
                    D_j = RolloutStorage(num_processes=self.config["num_processes"],
                                              state_dim=self.env.observation_space.shape,
                                              action_dim=1,  # Discrete value
                                              num_steps=self.config["policy"]["num_steps"],
                                              device=device
                                              )

                    self.sample_policy(inner_policy, self.proxy_oracles[j], self.proxy_oracle_models[j], num_steps=self.config["policy"]["num_steps"],
                                       policy_storage=D_j, density_penalty = True)  # Sample from policy[j]

                    inner_loss = self.adapt(D_j, inner_policy, diffopt)

                    logs[f"inner_loop/proxy/{j}/loop/{k}/loss/"] = inner_loss.item()
                    logs[f"inner_loop/policy/{j}/loop/{k}/action_logprob/"] = D_j.log_probs.mean().item()

                    train_episodes.append(D_j)
                    train_losses.append(inner_loss.detach())

                logs["timing/inner_loop_update"] = time.time() - time_st
                

                # Sample mols for meta update
                if self.config["outerloop"]["oracle"] == 'proxy':

                    self.sample_policy(inner_policy, self.proxy_oracles[j], self.proxy_oracle_models[j], num_steps=self.config["policy"]["num_meta_steps"],
                                       policy_storage=D_meta_query, density_penalty = self.config["outerloop"]["density_penalty"])
                elif self.config["outerloop"]["oracle"] == 'true':
                    self.sample_policy(inner_policy, self.true_oracle, self.true_oracle_model, num_steps=self.config["policy"]["num_meta_steps"],
                                                      policy_storage=D_meta_query, density_penalty = self.config["outerloop"]["density_penalty"])
                    # queried_mols = self.sample_policy(inner_policy, self.true_oracle, self.true_oracle_model, self.config["num_meta_proxy_samples"],
                    #                                   policy_storage=D_meta_query).detach()
                    # queried_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, queried_mols,
                    #                                                           flatten_input=self.flatten_true_oracle_input))

                    # self.D_train.insert(queried_mols, queried_mols_scores)

                    # # Sync query_history with D_train
                    # self.query_history += list(self.D_train.mols[len(self.query_history):self.D_train.storage_filled])
                else:
                    raise NotImplementedError


                # Save the episodes...

                episodes.append((train_episodes, D_meta_query))
                train_losses_s.append(train_losses)


        # Log the inner losses later (possibly instead of here)! 
        loss = self.step(episodes) # Performs meta-update
        logs["meta/loss"] = loss

        # self.meta_opt.step()
        return logs

    def sample_query_mols(self, logs, num_query_proxies, num_samples_per_proxy):


        # Proxy -- used for generating molecules for querying
        self.proxy_query_oracles = [AMPProxyOracle(training_storage=self.D_train, p=self.config["proxy_oracle"]["p"])
                                    for j in range(num_query_proxies)]
        self.proxy_query_oracle_models = [utl.get_proxy_oracle_model(self.config) for j in
                                          range(num_query_proxies)]
        st_time = time.time()
        # Sample molecules for Querying:
        # Fit proxy query oracles
        for j in range(num_query_proxies):
            self.proxy_query_oracle_models[j] = self.proxy_query_oracles[j].fit(
                self.proxy_query_oracle_models[j], flatten_input=self.flatten_proxy_oracle_input)

        logs["timing/fitting_proxy_query_oracle"] = time.time() - st_time
        st_time = time.time()

        inner_opt = optim.SGD(self.policy.parameters(), lr=self.config["inner_lr"])
        sampled_mols = []
        for j in range(num_query_proxies):
            # Proxy(Task)-specific updates
            with higher.innerloop_ctx(
                    self.policy, inner_opt, copy_initial_weights=False
            ) as (inner_policy, diffopt):

                for k in range(self.config["num_inner_updates"]):
                    D_j = RolloutStorage(num_processes=self.config["num_processes"],
                                              state_dim=self.env.observation_space.shape,
                                              action_dim=1,  # Discrete value
                                              num_steps=self.config["policy"]["num_steps"],
                                              device=device
                                              )

                    self.sample_policy(inner_policy, 
                                        oracle=self.proxy_query_oracles[j],
                                        oracle_model=self.proxy_query_oracle_models[j],
                                        num_steps=self.config["policy"]["num_steps"],
                                        num_samples=None,
                                        policy_storage=D_j,
                                        density_penalty = True)  # Sample from policy[j]


                    inner_loss = self.adapt(D_j, inner_policy, diffopt)

                    # Sample mols (and query) for training the proxy oracles later
                sampled_mols.append(self.sample_policy(inner_policy, self.proxy_query_oracles[j], self.proxy_query_oracle_models[j], 
                                                        num_steps=self.config["policy"]["num_steps"],
                                                        num_samples=num_samples_per_proxy).detach())  # Sample from policies -- preferably make this parallelised in the future

        return sampled_mols, logs

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
        data = {"reward_oracle": oracle,
                "reward_oracle_model": oracle_model,
                "query_history": self.query_history,
                "query_reward_in_env": query_reward and self.config["query_reward_in_env"], 
                "density_penalty": density_penalty}
        self.env.set_oracles(data)

        break_loop = False
        while not break_loop:
            """
                Loop if needed to generate more molecules to query...
            """


            state = torch.tensor(self.env.reset()).float()  # Returns (num_processes, 51, 21) -- TODO: ....this needs to be parallelised
            hidden_state = None
            for stepi in range(num_steps):

                masks = None
                st = state
                st = seq_to_encoding(st).flatten(-2, -1).to(device)  
                value, action, log_prob, hidden_state = policy.act(st, hidden_state, masks=masks)
                action = action.detach().cpu()

                next_state, reward, done, info = self.env.step(action) 


                done = torch.tensor(done).float()
                next_state = torch.tensor(next_state).float()
                reward = torch.tensor(reward)

                if policy_storage is not None:
                    policy_storage.insert(state=state.detach().clone(),
                                          next_state=next_state.detach().clone(),
                                          action=action.detach().clone(),
                                          reward=reward.detach().clone(),
                                          done=done.detach().clone())

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
            bool_idx = policy_storage.dones.bool()
            query_states = policy_storage.next_states[bool_idx]

            dens = diversity(query_states, 
                                self.query_history, 
                                div_switch=self.config["diversity"]["div_switch"], 
                                radius=self.config["env"]["radius"], 
                                div_metric_name=self.config["diversity"]["div_metric_name"]).get_density()            

            query_scores = oracle.query(oracle_model, query_states.cpu(), flatten_input=self.flatten_proxy_oracle_input)

            policy_storage.rewards[bool_idx] = torch.tensor(query_scores).to(device) - self.config["env"]["lambda"] * dens.to(device) # TODO: Set the rewards to include the density penalties...
            

        if policy_storage is not None:
            
            policy_storage.compute_returns()

            # TODO: If the meta optimiser doesn't use bootstrapping... then we do this
            policy_storage.after_rollouts()

        self.total_time_sampling += time.time() - time_st

        return return_mols

    def select_molecules(self, mols, logs):
        """
            Selects the molecules to query from the ones proposed by the policy trained on each of the proxy oracle
             - Let's select the ones that have the highest score according to the proxy oracles? -- Check...
        """

        # Remove duplicate molecules... in current batch
        mols = np.unique(mols, axis=0)

        # Remove duplicate molecules that have already been queried...
        valid_idx = []
        for i in range(mols.shape[0]):
            from data.dynappo_data import enc_to_seq
            seq = enc_to_seq(torch.tensor(mols[i]))
            seq = seq[:seq.find(">")]

            if seq not in self.D_train.mols_set:
                valid_idx.append(i)
            else:
                print("Already queried...")

        mols = mols[valid_idx] # Filter out the duplicates

        if mols.shape[0] > 0:
            mols = torch.tensor(mols)

            if self.iter_idx == 0:  # Special case: Random Policy
                n_query = self.config["num_initial_samples"]
            else:
                n_query = min(self.config["num_query_per_iter"], mols.shape[0])

            logs, scores = filtering.get_scores(self.config, mols, self.proxy_oracles, self.proxy_oracle_models, self.flatten_proxy_oracle_input, logs, iter_idx = self.iter_idx)

            _, sorted_idx = torch.sort(scores, descending = True)

            sorted_mols = mols.clone()[sorted_idx]
            selected_mols = filtering.select(self.config, sorted_mols, n_query)

            return selected_mols, logs
        else:
            return None, logs





    # Based off of CAVIA public code: 



    def adapt(self, episodes, policy, diffopt, first_order=False, params=None, lr=None):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """


        states = seq_to_encoding(episodes.states.flatten(0, 1)).view(episodes.states.shape).to(device)  

        # Fit the baseline to the training episodes
        self.baseline.fit(states, episodes.masks, episodes.returns)

        # Get the loss on the training episodes
        loss = self.inner_loss(episodes, policy)

        # Inner update
        diffopt.step(loss)

        return loss

    # def sample(self, tasks, first_order=False):
    #     """Sample trajectories (before and after the update of the parameters) 
    #     for all the tasks `tasks`.
    #     """
    #     episodes = []
    #     losses = []
    #     for task in tasks:
    #         self.sampler.reset_task(task)
    #         self.policy.reset_context()
    #         train_episodes = self.sampler.sample(self.policy, gamma=self.gamma)
    #         # inner loop (for CAVIA, this only updates the context parameters)
    #         params, loss = self.adapt(train_episodes, first_order=first_order)
    #         # rollouts after inner loop update
    #         valid_episodes = self.sampler.sample(self.policy, params=params, gamma=self.gamma)
    #         episodes.append((train_episodes, valid_episodes))
    #         losses.append(loss.item())

    #     return episodes, losses

    def kl_divergence(self, episodes, old_pis=None):
        inner_opt = optim.SGD(self.policy.parameters(), lr=self.config["inner_lr"])
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes_s, valid_episodes), old_pi in zip(episodes, old_pis):

            with higher.innerloop_ctx(
                    self.policy, inner_opt, copy_initial_weights=False
            ) as (inner_policy, diffopt):

                for train_episodes in train_episodes_s:
                    self.adapt(train_episodes, inner_policy, diffopt)

                # this is the inner-loop update
                st = seq_to_encoding(valid_episodes.states.flatten(0, 1)).flatten(-2, -1).view(valid_episodes.states.shape[0], valid_episodes.states.shape[1], -1).to(device)  
                _, _, _, _, pi = inner_policy.act(st, None, None, return_dist=True)

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=valid_episodes.masks)
                kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""

        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        inner_opt = optim.SGD(self.policy.parameters(), lr=self.config["inner_lr"])

        losses, kls, pis = [], [], []

        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes_s, valid_episodes), old_pi in zip(episodes, old_pis):

            with higher.innerloop_ctx(
                    self.policy, inner_opt, copy_initial_weights=False
            ) as (inner_policy, diffopt):

                for train_episodes in train_episodes_s:
                    self.adapt(train_episodes, inner_policy, diffopt)

                with torch.set_grad_enabled(old_pi is None):

                    # get action values after inner-loop update

                    st = seq_to_encoding(valid_episodes.states.flatten(0, 1)).flatten(-2, -1).view(valid_episodes.states.shape[0], valid_episodes.states.shape[1], -1).to(device)  
                    _, _, _, _, pi = inner_policy.act(st, None, None, return_dist=True) # Fix the policy..., so it returns "pi"
                    pis.append(detach_distribution(pi))

                    if old_pi is None:
                        old_pi = detach_distribution(pi)

                    st = st.view(valid_episodes.states.shape)
                    values = self.baseline(states=st, returns=valid_episodes.returns, masks=valid_episodes.masks) # To fix this too...
                    advantages = valid_episodes.gae(values, tau=self.config["metalearner"]["tau"], gamma=self.config["metalearner"]["gamma"])
                    advantages = weighted_normalize(advantages, weights=valid_episodes.masks) # masks -> mask

                    log_ratio = (pi.log_prob(valid_episodes.actions.squeeze(-1))
                                 - old_pi.log_prob(valid_episodes.actions.squeeze(-1)))
                    # if log_ratio.dim() > 2:
                    #     log_ratio = torch.sum(log_ratio, dim=2)
                    ratio = torch.exp(log_ratio)


                    loss = -weighted_mean(ratio * advantages, dim=0, weights=valid_episodes.masks)
                    losses.append(loss)

                    kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=valid_episodes.masks)
                    kls.append(kl)

        return torch.mean(torch.stack(losses, dim=0)), torch.mean(torch.stack(kls, dim=0)), pis

    def step(self, episodes):
        """Meta-optimization step (ie. update of the initial parameters)
        """

        if self.config["metalearner"]["method"] == "REINFORCE":
            return self.reinforce_step(episodes)
        elif self.config["metalearner"]["method"] == "TRPO":
            return self.trpo_step(episodes)
        else:
            raise NotImplementedError

    def reinforce_step(self, episodes):
        inner_opt = optim.SGD(self.policy.parameters(), lr=self.config["inner_lr"])

        self.meta_opt.zero_grad()
        meta_losses = []
        
        for (train_episodes_s, valid_episodes) in episodes:

            with higher.innerloop_ctx(
                    self.policy, inner_opt, copy_initial_weights=False
            ) as (inner_policy, diffopt):

                for train_episodes in train_episodes_s:
                    self.adapt(train_episodes, inner_policy, diffopt)

                # get action values after inner-loop update
                st = seq_to_encoding(valid_episodes.states.flatten(0, 1)).flatten(-2, -1).view(valid_episodes.states.shape[0], valid_episodes.states.shape[1], -1).to(device)  
                _, _, _, _, pi = inner_policy.act(st, None, None, return_dist=True) # Fix the policy..., so it returns "pi"

                log_prob = pi.log_prob(valid_episodes.actions.squeeze(-1))


                loss = -weighted_mean(log_prob * valid_episodes.returns, dim=0, weights=valid_episodes.masks)

                meta_losses.append(loss)


        # for (_, valid_episode) in episodes:
            
        #     reinforce_loss = rl_utl.reinforce_loss(valid_episode)
        #     entropy_bonus = self.config["entropy_reg_coeff"] * rl_utl.entropy_bonus(valid_episode)
        #     meta_loss = reinforce_loss + entropy_bonus

        #     meta_losses.append(meta_loss)

        # loss = sum(meta_losses)/len(meta_losses)
        loss = sum(meta_losses)

        loss_item = loss.item()

        loss.backward()
        self.meta_opt.step()
        
        return loss_item 

    def trpo_step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        old_loss, _, old_pis = self.surrogate_loss(episodes)
        # this part will take higher order gradients through the inner loop:
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes, damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 10.0 # Leo: Originally 1.0...
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step, self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            print('no update?')
            vector_to_parameters(old_params, self.policy.parameters())

        return loss




    #  ============ LOGGING BELOW







    def log(self, logs):
        """
            Tentatively Tensorboard, but perhaps we want WANDB
        """
        num_queried = self.config["num_initial_samples"] + self.iter_idx * self.config["num_query_per_iter"]

        # log the average weights and gradients of all models (where applicable)
        for [model, name] in [
            [self.policy, 'policy']
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
