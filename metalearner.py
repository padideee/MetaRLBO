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
from oracles.proxy.AMP_proxy_oracle import AMPProxyOracle

from environments.AMP_env import AMPEnv
from environments.parallel_envs import make_vec_envs
import higher
from utils.tb_logger import TBLogger

from evaluation import get_test_oracle
from data.process_data import seq_to_encoding
from algo.diversity import pairwise_hamming_distance
import time

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
                                              "div_switch"]) # TODO: This diversity switch should be implemented for the other envs too
                            

        self.D_train = QueryStorage(storage_size=self.config["query_storage_size"],
                                    state_dim=self.env.observation_space.shape)

        self.query_history = []
        # Proxy -- used for training
        self.proxy_oracles = [AMPProxyOracle(training_storage=self.D_train, p=self.config["proxy_oracle"]["p"]) for j in
                              range(self.config["num_proxies"])]
        self.proxy_oracle_models = [utl.get_proxy_oracle_model(self.config) for j in range(self.config["num_proxies"])]



        # Proxy -- used for generating molecules for querying
        self.proxy_query_oracles = [AMPProxyOracle(training_storage=self.D_train, p=self.config["proxy_oracle"]["p"])
                                    for j in range(self.config["num_query_proxies"])]
        self.proxy_query_oracle_models = [utl.get_proxy_oracle_model(self.config) for j in
                                          range(self.config["num_query_proxies"])]

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

        self.meta_opt = optim.SGD(self.policy.parameters(), lr=self.config["outer_lr"])
        self.test_oracle = get_test_oracle()
        self.iter_idx = 0


    def run(self):

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
                    "num_initial_samples"] * 2)  # Sample from true env. using random policy (num_starting_mols, dim of mol)

            else:
                # Training
                for mi in range(self.config["num_meta_updates_per_iter"]):
                    logs = self.meta_update(logs)

                # Sample molecules for Querying:
                # Fit proxy query oracles
                for j in range(self.config["num_query_proxies"]):
                    self.proxy_query_oracle_models[j] = self.proxy_query_oracles[j].fit(
                        self.proxy_query_oracle_models[j], flatten_input=self.flatten_proxy_oracle_input)

                sampled_mols, logs = self.sample_query_mols(logs)

                # Perform the querying stage
                # This is a bug...
                sampled_mols = torch.cat(sampled_mols, dim=0)

            # Do some filtering of the molecules here...
            queried_mols, logs = self.select_molecules(sampled_mols, logs)

            # Perform the querying
            if queried_mols is not None:
                # Query the scores
                queried_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, queried_mols,
                                                                          flatten_input=self.flatten_true_oracle_input))

                self.D_train.insert(queried_mols, queried_mols_scores)

                # Sync query_history with D_train
                self.query_history += list(self.D_train.mols[len(self.query_history):self.D_train.storage_filled])

                logs["outer_loop/queried_mols_scores/current_batch/mean"] = queried_mols_scores.mean().item()
                logs["outer_loop/queried_mols_scores/current_batch/max"] = queried_mols_scores.max().item()

                # TODO: Log diversity here... parallelise the querying (after the unique checking)
                logs["outer_loop/queried_mols/diversity"] = pairwise_hamming_distance(queried_mols)  # TODO

            logs[f"outer_loop/sampled_mols_scores/cumulative/mean"] = self.D_train.scores[
                                                                      :self.D_train.storage_filled].mean().item()
            logs[f"outer_loop/sampled_mols_scores/cumulative/max"] = self.D_train.scores[
                                                                     :self.D_train.storage_filled].max().item()
            logs[f"outer_loop/sampled_mols_scores/cumulative/min"] = self.D_train.scores[
                                                                     :self.D_train.storage_filled].min().item()

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

                self.log(logs)

                utl.save_mols(mols=self.D_train.mols[:self.D_train.storage_filled].numpy(),
                              scores=self.D_train.scores[:self.D_train.storage_filled].numpy(),
                              folder=self.logger.full_output_folder)

    def meta_update(self, logs):

        inner_opt = optim.SGD(self.policy.parameters(), lr=self.config["inner_lr"])
        self.meta_opt.zero_grad()

        # Fit proxy oracles
        for j in range(self.config["num_proxies"]):
            self.proxy_oracle_models[j] = self.proxy_oracles[j].fit(self.proxy_oracle_models[j],
                                                                    flatten_input=self.flatten_proxy_oracle_input)

        # Proxy(Task)-specific updates
        for j in range(self.config["num_proxies"]):
            with higher.innerloop_ctx(
                    self.policy, inner_opt, copy_initial_weights=False
            ) as (inner_policy, diffopt):

                self.D_meta_query = RolloutStorage(num_samples=self.config["num_meta_proxy_samples"],
                                                   state_dim=self.env.observation_space.shape,
                                                   action_dim=1,  # Discrete value
                                                   hidden_dim=self.config["policy"]["hidden_dim"] if "hidden_dim" in
                                                                                                     self.config[
                                                                                                         "policy"] else None,
                                                   num_steps=self.config["policy"]["num_steps"],
                                                   device=device
                                                   )

                for k in range(self.config["num_inner_updates"]):
                    self.D_j = RolloutStorage(num_samples=self.config["num_samples_per_task_update"],
                                              state_dim=self.env.observation_space.shape,
                                              action_dim=1,  # Discrete value
                                              hidden_dim=self.config["policy"]["hidden_dim"] if "hidden_dim" in
                                                                                                self.config[
                                                                                                    "policy"] else None,
                                              num_steps=self.config["policy"]["num_steps"],
                                              device=device
                                              )

                    self.sample_policy(inner_policy, self.proxy_oracles[j], self.proxy_oracle_models[j], self.config["num_samples_per_task_update"],
                                       policy_storage=self.D_j)  # Sample from policy[j]

                    self.D_j.compute_log_probs(inner_policy)
                    inner_loss = rl_utl.reinforce_loss(self.D_j) + self.config[
                        "entropy_reg_coeff"] * rl_utl.entropy_bonus(self.D_j)

                    logs[f"inner_loop/proxy/{j}/loop/{k}/loss/"] = inner_loss.item()
                    logs[f"inner_loop/policy/{j}/loop/{k}/action_logprob/"] = self.D_j.log_probs.mean().item()

                    # Inner update
                    diffopt.step(inner_loss)

                    # Sample mols for meta update
                if self.config["outerloop_oracle"] == 'proxy':

                    self.sample_policy(inner_policy, self.proxy_oracles[j], self.proxy_oracle_models[j], self.config["num_meta_proxy_samples"],
                                       policy_storage=self.D_meta_query)
                elif self.config["outerloop_oracle"] == 'true':
                    queried_mols = self.sample_policy(inner_policy, true_oracle, true_oracle_model, self.config["num_meta_proxy_samples"],
                                                      policy_storage=self.D_meta_query).detach()
                    queried_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, queried_mols,
                                                                              flatten_input=self.flatten_true_oracle_input))

                    self.D_train.insert(queried_mols, queried_mols_scores)

                    # Sync query_history with D_train
                    self.query_history += list(self.D_train.mols[len(self.query_history):self.D_train.storage_filled])
                else:
                    raise NotImplementedError


                reinforce_loss = rl_utl.reinforce_loss(self.D_meta_query)
                entropy_bonus = self.config["entropy_reg_coeff"] * rl_utl.entropy_bonus(self.D_meta_query)
                meta_loss = reinforce_loss + entropy_bonus
                logs["meta/reinforce_loss"] = reinforce_loss
                logs["meta/entropy_bonus"] = entropy_bonus

                meta_loss.backward()

        self.meta_opt.step()
        return logs

    def sample_query_mols(self, logs):
        inner_opt = optim.SGD(self.policy.parameters(), lr=self.config["inner_lr"])
        sampled_mols = []
        for j in range(self.config["num_query_proxies"]):
            # Proxy(Task)-specific updates
            with higher.innerloop_ctx(
                    self.policy, inner_opt, copy_initial_weights=False
            ) as (inner_policy, diffopt):

                self.D_meta_query = RolloutStorage(num_samples=self.config["num_meta_proxy_samples"],
                                                   state_dim=self.env.observation_space.shape,
                                                   action_dim=1,  # Discrete value
                                                   hidden_dim=self.config["policy"]["hidden_dim"] if "hidden_dim" in
                                                                                                     self.config[
                                                                                                         "policy"] else None,
                                                   num_steps=self.config["policy"]["num_steps"],
                                                   device=device
                                                   )

                for k in range(self.config["num_inner_updates"]):
                    self.D_j = RolloutStorage(num_samples=self.config["num_samples_per_task_update"],
                                              state_dim=self.env.observation_space.shape,
                                              action_dim=1,  # Discrete value
                                              hidden_dim=self.config["policy"]["hidden_dim"] if "hidden_dim" in
                                                                                                self.config[
                                                                                                    "policy"] else None,
                                              num_steps=self.config["policy"]["num_steps"],
                                              device=device
                                              )

                    self.sample_policy(inner_policy, 
                                        oracle=self.proxy_query_oracles[j],
                                        oracle_model=self.proxy_query_oracle_models[j],
                                        num_samples=self.config["num_samples_per_task_update"],
                                        policy_storage=self.D_j)  # Sample from policy[j]

                    self.D_j.compute_log_probs(inner_policy)
                    inner_loss = rl_utl.reinforce_loss(self.D_j) + self.config[
                        "entropy_reg_coeff"] * rl_utl.entropy_bonus(self.D_j)

                    logs[f"query/inner_loop/proxy/{j}/loop/{k}/loss/"] = inner_loss.item()
                    logs[f"query/inner_loop/policy/{j}/loop/{k}/action_logprob/"] = self.D_j.log_probs.mean().item()

                    # Inner update
                    diffopt.step(inner_loss)

                    # Sample mols (and query) for training the proxy oracles later
                sampled_mols.append(self.sample_policy(inner_policy, self.proxy_query_oracles[j], self.proxy_query_oracle_models[j], self.config[
                    "num_samples_per_iter"]).detach())  # Sample from policies -- preferably make this parallelised in the future

        return sampled_mols, logs

    def sample_policy(self, policy, oracle, oracle_model, num_samples, policy_storage=None):
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

        # TODO: Include oracle, oracle_model, query_history into the env.step...
        state_dim = self.env.observation_space.shape
        return_mols = torch.zeros(num_samples, *state_dim)

        curr_sample = 0
        # while (policy_storage is None and curr_queried < num_samples) or (
        #         policy_storage is not None and curr_sample < num_samples):
        data = {"reward_oracle": oracle,
                "reward_oracle_model": oracle_model,
                "query_history": self.query_history,
                "query_reward": policy_storage is not None}
        self.env.set_oracles(data)
        for meta_update in range((num_samples - 1) // self.config["num_processes"] + 1):
            """
                Either policy_storage is None, meaning return queried molecules
                Or policy_storage is not None, meaning fill up the storage
            """

            state = torch.tensor(self.env.reset()).float()  # Returns (num_processes, 51, 21) -- TODO: ....this needs to be parallelised
            hidden_state = None
            for stepi in range(self.config["policy"]["num_steps"]):

                masks = None
                st = state
                st = seq_to_encoding(st).flatten(-2, -1).to(device)  
                # if self.iter_idx != 0:
                #     import pdb; pdb.set_trace()
                value, action, log_prob, hidden_state = policy.act(st, hidden_state, masks=masks)
                action = action.detach().cpu()

                next_state, reward, done, info = self.env.step(action) 


                done = torch.tensor(done).float().unsqueeze(-1)
                next_state = torch.tensor(next_state).float()
                reward = torch.tensor(reward).unsqueeze(-1)
                # if done: # TODO: setup the done...
                #     return_mols[
                #         curr_queried] = next_state.detach().clone()  # Leo: The return mols is bugged if we're using the "true oracle" to perform the meta-update....
                #     curr_queried += 1

                # 2 choices: Either we forcibly generate the molecules (cut off at 50 or we let it generate until then...)

                if policy_storage is not None:
                    policy_storage.insert(state=state.detach().clone(),
                                          next_state=next_state.detach().clone(),
                                          action=action.detach().clone(),
                                          reward=reward.detach().clone(),
                                          log_prob=log_prob.clone(),
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
                                return return_mols # Return if sufficient number of molecules are found

                    next_state = utl.reset_env(self.env, self.config, done_indices, next_state)


                state = next_state.clone()

            if policy_storage is not None:
                policy_storage.after_traj(incr = self.config["num_processes"])

        if policy_storage is not None:
            
            policy_storage.compute_returns()

            # import pdb; pdb.set_trace()
            # TODO: If the meta optimiser doesn't use bootstrapping... then we do this
            policy_storage.after_rollouts()

            # assert policy_storage.dones.sum() == num_samples

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
            tuple_mol = tuple(mols[i].flatten().tolist())
            if tuple_mol not in self.D_train.mols_set:
                valid_idx.append(i)
            else:
                print("Already queried...")
        mols = mols[valid_idx]

        if mols.shape[0] > 0:
            mols = torch.tensor(mols)

            if self.iter_idx == 0:  # Special case: Random Policy
                n_query = self.config["num_initial_samples"]
            else:
                n_query = min(self.config["num_query_per_iter"], mols.shape[0])

            if self.config["selection_criteria"]["method"] == "RANDOM" or self.iter_idx == 0:
                # In the case that the iter_idx is 0, then we can only randomly select them...
                perm = torch.randperm(mols.shape[0])
                idx = perm[:n_query]
            elif self.config["selection_criteria"]["method"] == "PROXY_MEAN":
                proxy_scores = []
                for j in range(self.config["num_proxies"]):
                    proxy_scores.append(torch.tensor(self.proxy_oracles[j].query(self.proxy_oracle_models[j], mols,
                                                                                 flatten_input=self.flatten_proxy_oracle_input)))
                proxy_scores = torch.stack(proxy_scores)
                proxy_scores_mean = proxy_scores.mean(dim=0)

                _, sorted_idx = torch.sort(proxy_scores_mean)

                sorted_idx = torch.flip(sorted_idx, dims=(0,))  # Largest to Smallest

                idx = sorted_idx[:n_query]  # Select top scores
            elif self.config["selection_criteria"]["method"] == "UCB":
                proxy_scores = []
                for j in range(self.config["num_proxies"]):
                    proxy_scores.append(torch.tensor(self.proxy_oracles[j].query(self.proxy_oracle_models[j], mols,
                                                                                 flatten_input=self.flatten_proxy_oracle_input)))
                proxy_scores = torch.stack(proxy_scores)
                proxy_scores_mean = proxy_scores.mean(dim=0)

                proxy_scores_std = proxy_scores.std(dim=0)

                logs["select_molecules/proxy_model/mean/mean"] = proxy_scores_mean.mean()
                logs["select_molecules/proxy_model/std/mean"] = proxy_scores_std.mean()

                scores = proxy_scores_mean + self.config["selection_criteria"]["config"]["beta"] * proxy_scores_std
                _, sorted_idx = torch.sort(scores)

                sorted_idx = torch.flip(sorted_idx, dims=(0,))  # Largest to Smallest

                idx = sorted_idx[:n_query]  # Select top scores

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