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
        
        self.env = AMPEnv(self.true_oracle, self.true_oracle_model, lambd=self.config["env"]["lambda"], radius = self.config["env"]["radius"]) # The reward will not be needed in this env.

        self.D_train = QueryStorage(storage_size=self.config["query_storage_size"], state_dim = self.env.observation_space.shape)


        self.query_history = []
        # Proxy -- used for training
        self.proxy_oracles = [AMPProxyOracle(training_storage=self.D_train, p=self.config["proxy_oracle"]["p"]) for j in range(self.config["num_proxies"])]
        self.proxy_oracle_models = [utl.get_proxy_oracle_model(self.config) for j in range(self.config["num_proxies"])]
        self.proxy_envs = [AMPEnv(self.proxy_oracles[j], self.proxy_oracle_models[j], lambd=self.config["env"]["lambda"], query_history = self.query_history) for j in range(self.config["num_proxies"])]


        # Proxy -- used for generating molecules for querying
        self.proxy_query_oracles = [AMPProxyOracle(training_storage=self.D_train, p=self.config["proxy_oracle"]["p"]) for j in range(self.config["num_query_proxies"])]
        self.proxy_query_oracle_models = [utl.get_proxy_oracle_model(self.config) for j in range(self.config["num_query_proxies"])]
        self.proxy_query_envs = [AMPEnv(self.proxy_query_oracles[j], self.proxy_query_oracle_models[j], lambd=self.config["env"]["lambda"], query_history = self.query_history) for j in range(self.config["num_query_proxies"])]

        # We need to include the molecules... in terms of diversity. 
        # We need to add the model to the environment... in order to query.
        # We need to fix the query function in the environment...

        if self.config["policy"]["model_name"] == "GRU":
            self.policy = CategoricalGRUPolicy(num_actions = self.env.action_space.n + 1,
                                                hidden_size = self.config["policy"]["model_config"]["hidden_dim"],
                                                state_dim = self.env.action_space.n + 1,
                                                state_embed_dim = self.config["policy"]["model_config"]["state_embedding_size"],
                                                ).to(device)
        elif self.config["policy"]["model_name"] == "MLP":
            from policies.mlp_policy import Policy as MLPPolicy
            self.policy = MLPPolicy((self.env.observation_space.shape[0] * self.env.observation_space.shape[1],),
                                   self.env.action_space).to(device)
        elif self.config["policy"]["model_name"] == "RANDOM":
            self.policy = RandomPolicy(input_size = self.env.observation_space.shape, output_size = 1, num_actions=self.env.action_space.n).to(device)
        else:
            raise NotImplementedError


        self.meta_opt = optim.SGD(self.policy.parameters(), lr=self.config["outer_lr"])
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
            if self.iter_idx == 0:

                random_policy = RandomPolicy(input_size = self.env.observation_space.shape, output_size = 1, num_actions=self.env.action_space.n).to(device)
                sampled_mols = self.sample_policy(random_policy, self.env, self.config["num_initial_samples"]) # Sample from true env. using random policy (num_starting_mols, dim of mol)

                sampled_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, sampled_mols, flatten_input = self.flatten_true_oracle_input))
                #[Prob. False, Prob. True]

                self.D_train.insert(sampled_mols, sampled_mols_scores) 

                # Sync query_history with D_train
                self.query_history += list(self.D_train.mols[len(self.query_history):self.D_train.storage_filled])




            # Training
            for mi in range(self.config["num_meta_updates_per_iter"]):
                logs = self.meta_update(logs)



            # Sample molecules for Querying:
            # Fit proxy query oracles
            for j in range(self.config["num_query_proxies"]):
                self.proxy_query_oracle_models[j] = self.proxy_query_oracles[j].fit(self.proxy_query_oracle_models[j], flatten_input = self.flatten_proxy_oracle_input)


            sampled_mols, logs = self.sample_query_mols(logs)

            # Perform the querying stage
            # This is a bug...
            sampled_mols = torch.cat(sampled_mols, dim = 0)

            # Do some filtering of the molecules here...
            queried_mols = self.select_molecules(sampled_mols)


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
            self.proxy_oracle_models[j] = self.proxy_oracles[j].fit(self.proxy_oracle_models[j], flatten_input = self.flatten_proxy_oracle_input)


        # Proxy(Task)-specific updates
        for j in range(self.config["num_proxies"]):
            with higher.innerloop_ctx(
                    self.policy, inner_opt, copy_initial_weights=False
            ) as (inner_policy, diffopt):

                self.D_meta_query = RolloutStorage(num_samples = self.config["num_meta_proxy_samples"],
                                                   state_dim = self.env.observation_space.shape,
                                                   action_dim = 1, # Discrete value
                                                   hidden_dim = self.config["policy"]["hidden_dim"] if "hidden_dim" in self.config["policy"] else None,
                                                   num_steps = self.env.max_AMP_length,
                                                   device = device
                                                   )

                for k in range(self.config["num_inner_updates"]):
                    self.D_j = RolloutStorage(num_samples = self.config["num_samples_per_task_update"],
                                               state_dim = self.env.observation_space.shape,
                                               action_dim = 1, # Discrete value
                                               hidden_dim = self.config["policy"]["hidden_dim"] if "hidden_dim" in self.config["policy"] else None,
                                               num_steps = self.env.max_AMP_length,
                                               device=device
                                               )
                    


                    self.sample_policy(inner_policy, self.proxy_envs[j], self.config["num_samples_per_task_update"], policy_storage=self.D_j) # Sample from policy[j]

                    self.D_j.compute_log_probs(inner_policy)
                    inner_loss = rl_utl.reinforce_loss(self.D_j) 

                    logs[f"inner_loop/proxy/{j}/loop/{k}/loss/"] = inner_loss.item()
                    logs[f"inner_loop/policy/{j}/loop/{k}/action_logprob/"] = self.D_j.log_probs.mean().item()
                    

                    # Inner update
                    diffopt.step(inner_loss) 




                # Sample mols for meta update
                if self.config["outerloop_oracle"] == 'proxy':

                    self.sample_policy(inner_policy, self.proxy_envs[j], self.config["num_meta_proxy_samples"], policy_storage=self.D_meta_query) 
                elif self.config["outerloop_oracle"] == 'true':
                    queried_mols = self.sample_policy(inner_policy, self.env, self.config["num_meta_proxy_samples"], policy_storage=self.D_meta_query).detach()
                    queried_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, queried_mols, flatten_input = self.flatten_true_oracle_input))

                    self.D_train.insert(queried_mols, queried_mols_scores) 

                    # Sync query_history with D_train
                    self.query_history += list(self.D_train.mols[len(self.query_history):self.D_train.storage_filled])
                else:
                    raise NotImplementedError




                meta_loss = rl_utl.reinforce_loss(self.D_meta_query)
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

                self.D_meta_query = RolloutStorage(num_samples = self.config["num_meta_proxy_samples"],
                                                   state_dim = self.env.observation_space.shape,
                                                   action_dim = 1, # Discrete value
                                                   hidden_dim = self.config["policy"]["hidden_dim"] if "hidden_dim" in self.config["policy"] else None,
                                                   num_steps = self.env.max_AMP_length,
                                                   device = device
                                                   )

                for k in range(self.config["num_inner_updates"]):
                    self.D_j = RolloutStorage(num_samples = self.config["num_samples_per_task_update"],
                                               state_dim = self.env.observation_space.shape,
                                               action_dim = 1, # Discrete value
                                               hidden_dim = self.config["policy"]["hidden_dim"] if "hidden_dim" in self.config["policy"] else None,
                                               num_steps = self.env.max_AMP_length,
                                               device=device
                                               )
                    


                    self.sample_policy(inner_policy, self.proxy_query_envs[j], self.config["num_samples_per_task_update"], policy_storage=self.D_j) # Sample from policy[j]

                    self.D_j.compute_log_probs(inner_policy)
                    inner_loss = rl_utl.reinforce_loss(self.D_j) 

                    logs[f"query/inner_loop/proxy/{j}/loop/{k}/loss/"] = inner_loss.item()
                    logs[f"query/inner_loop/policy/{j}/loop/{k}/action_logprob/"] = self.D_j.log_probs.mean().item()
                    

                    # Inner update
                    diffopt.step(inner_loss) 

                # Sample mols (and query) for training the proxy oracles later
                sampled_mols.append(self.sample_policy(inner_policy, self.env, self.config["num_samples_per_iter"]).detach()) # Sample from policies -- preferably make this parallelised in the future


        return sampled_mols, logs




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
        state_dim = env.observation_space.shape # Currently hardcoded
        return_mols = torch.zeros(num_samples, *state_dim)

        curr_sample = 0
        curr_queried = 0
        while (policy_storage is None and curr_queried < num_samples) or (policy_storage is not None and curr_sample < num_samples):
            """
                Either policy_storage is None, meaning return queried molecules
                Or policy_storage is not None, meaning fill up the storage
            """


            end_traj = False
            state = env.reset().clone() # Returns (51, 21)
            hidden_state = None
            curr_timestep = 0
            curr_sample += 1
            while not end_traj:
                
                if self.config["policy"]["model_name"] == "GRU":
                    if curr_timestep == 0:
                        s = torch.zeros(state[0].shape)
                    else:
                        s = state[curr_timestep - 1]
                    action, log_prob, hidden_state = policy.act(s, hidden_state) 
                else:
                    masks = None
                    s = state.float().unsqueeze(0) # batch size is 1
                    s = seq_to_encoding(s).flatten(-2, -1).to(device) # batch size is 1 and positional encoding
                    
                    value, action, log_prob, hidden_state = policy.act(s, hidden_state, masks=masks)

                    action = action[0].detach() # batch size = 1
                    log_prob = log_prob[0] # batch size = 1



                next_state, reward, pred_prob, done, info = env.step(action, query_reward = policy_storage is not None)

                if done:
                    return_mols[curr_queried] = next_state.detach().clone() # Leo: The return mols is bugged if we're using the "true oracle" to perform the meta-update....
                    curr_queried += 1
                    end_traj = True


                # 2 choices: Either we forcibly generate the molecules (cut off at 50 or we let it generate until then...)

                if policy_storage is not None:
                    policy_storage.insert(state=state.detach().clone(), 
                                   next_state=next_state.detach().clone(),
                                   action=action.detach().clone(), 
                                   reward=reward.detach().clone(),
                                   log_prob=log_prob.clone(),
                                   done=torch.tensor(done).detach().float())

                state = next_state.clone()
                curr_timestep += 1

            if policy_storage is not None:
                policy_storage.compute_returns()
                policy_storage.after_rollout()

        return return_mols



    def select_molecules(self, mols):
        """
            Selects the molecules to query from the ones proposed by the policy trained on each of the proxy oracle


             - Let's select the ones that have the highest score according to the proxy oracles? -- Check...
        """


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

            n_query = min(self.config["num_query_per_iter"], mols.shape[0])
            
            if self.config["select_samples"]["method"] == "RANDOM":
                perm = torch.randperm(mols.shape[0])
                idx = perm[:n_query]
            elif self.config["select_samples"]["method"] == "PROXY_MEAN":
                proxy_scores = []
                for j in range(self.config["num_proxies"]):
                    proxy_scores.append(torch.tensor(self.proxy_oracles[j].query(self.proxy_oracle_models[j], mols, flatten_input = self.flatten_proxy_oracle_input)))
                proxy_scores_mean = sum(proxy_scores) / self.config["num_proxies"]
                
                _, sorted_idx = torch.sort(proxy_scores_mean)

                sorted_idx = torch.flip(sorted_idx, dims=(0,)) # Largest to Smallest
                
                idx = sorted_idx[:n_query] # Select top scores
            else:
                raise NotImplementedError

            selected_mols = mols.clone()[idx]

            # TODO: Filter the duplicate molecules... so there's no duplicates between the ones being queried and the "query_history..."
            return selected_mols

        else:
            return None






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
                    param_list_grad = list(filter(lambda x : x is not None, param_list_grad))
                    
                    param_grad_mean = np.mean(
                        [param_list_grad[i].cpu().numpy().mean() for i in range(len(param_list_grad))])

                    self.logger.add('gradients/{}'.format(name), param_grad_mean, num_queried)

        for k, v in logs.items():
            self.logger.add(k, v, num_queried)
