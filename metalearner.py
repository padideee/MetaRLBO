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
from data.process_data import get_AMP_data
import higher 
from utils.tb_logger import TBLogger

from evaluation import get_test_proxy
from data.process_data import seq_to_encoding




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
        D_AMP = get_AMP_data('data/data_train.hkl')
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
        
        self.env = AMPEnv(self.true_oracle_model, lambd=self.config["env"]["lambda"], radius = self.config["env"]["radius"]) # The reward will not be needed in this env.

        self.D_train = QueryStorage(storage_size=self.config["max_num_queries"], state_dim = self.env.observation_space.shape)


        self.query_history = []
        self.proxy_oracles = [AMPProxyOracle(training_storage=self.D_train, p=self.config["proxy_oracle"]["p"]) for j in range(self.config["num_proxies"])]
        self.proxy_oracle_models = [utl.get_proxy_oracle_model(self.config) for j in range(self.config["num_proxies"])]
        self.proxy_envs = [AMPEnv(self.proxy_oracle_models[j], lambd=self.config["env"]["lambda"], query_history = self.query_history) for j in range(self.config["num_proxies"])]


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
        else:
            raise NotImplementedError


        self.iter_idx = 0


    def run(self):

        self.true_oracle_model = self.true_oracle.fit(self.true_oracle_model, flatten_input = self.flatten_true_oracle_input)
        
        # updated_params = [None for _ in range(self.config["num_proxies"])]

        for self.iter_idx in tqdm(range(self.config["num_meta_updates"])):

            logs = {} 


            # Sample molecules to train proxy oracles
            if self.iter_idx == 0:

                random_policy = RandomPolicy(input_size = self.env.observation_space.shape, output_size = 1, num_actions=self.env.action_space.n).to(device)
                sampled_mols = self.sample_policy(random_policy, self.env, self.config["num_initial_samples"]) # Sample from true env. using random policy (num_starting_mols, dim of mol)
                

                self.query_history += [sampled_mols[i] for i in range(sampled_mols.shape[0])]


                sampled_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, sampled_mols, flatten_input = self.flatten_true_oracle_input))
                #[Prob. False, Prob. True]

                self.D_train.insert(sampled_mols, sampled_mols_scores[:, 1]) 

            # Fit proxy oracles
            for j in range(self.config["num_proxies"]):
                self.proxy_oracle_models[j] = self.proxy_oracles[j].fit(self.proxy_oracle_models[j], flatten_input = self.flatten_proxy_oracle_input)


            inner_opt = optim.SGD(self.policy.parameters(), lr=self.config["inner_lr"])
            meta_opt = optim.SGD(self.policy.parameters(), lr=self.config["outer_lr"])
            meta_opt.zero_grad()
            meta_losses = []
            meta_scores = []

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
                    self.sample_policy(inner_policy, self.proxy_envs[j], self.config["num_meta_proxy_samples"], policy_storage=self.D_meta_query) # Sample from policies using (update_params)




                    # Sample mols (and query) for training the proxy oracles later
                    sampled_mols = self.sample_policy(inner_policy, self.env, self.config["num_samples_per_iter"]).detach() # Sample from policies -- preferably make this parallelised in the future
                    
                    self.query_history += [sampled_mols[i] for i in range(sampled_mols.shape[0])]

                    # Query the scores
                    sampled_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, sampled_mols, flatten_input = self.flatten_true_oracle_input))[:, 1]
                    meta_scores.append(sampled_mols_scores)

                    logs[f"inner_loop/proxy/{j}/sampled_mols_scores/mean"] = sampled_mols_scores.mean().item()
                    logs[f"inner_loop/proxy/{j}/sampled_mols_scores/max"] = sampled_mols_scores.max().item()
                    logs[f"inner_loop/proxy/{j}/sampled_mols_scores/min"] = sampled_mols_scores.min().item()

                    topk_values, _ = sampled_mols_scores.topk(self.config["logging"]["top-k"])
                    logs[f"inner_loop/proxy/{j}/sampled_mols_scores/top-k/mean"] = topk_values.mean().item()

                    self.D_train.insert(sampled_mols, sampled_mols_scores)


                    meta_loss = rl_utl.reinforce_loss(self.D_meta_query)
                    meta_losses.append(meta_loss.detach())
                    meta_loss.backward() 


            print(len(self.query_history), len(self.proxy_envs[0].history))
            outer_loss = sum(meta_losses) / self.config["num_meta_proxy_samples"] / self.config["num_proxies"]
            outer_score = sum(meta_scores).sum() / self.config["num_meta_proxy_samples"] / self.config["num_proxies"]
            print(f"Outer Loss: {outer_loss}")
            print(f"Outer Scores: {outer_score}")
            logs["outer_loop/loss"] = outer_loss.item()
            logs["outer_loop/score"] = outer_score.item()
            meta_opt.step()

            logs[f"outer_loop/sampled_mols_scores/cumulative/mean"] = self.D_train.scores[:self.D_train.storage_filled].mean().item()
            logs[f"outer_loop/sampled_mols_scores/cumulative/max"] = self.D_train.scores[:self.D_train.storage_filled].max().item() 
            logs[f"outer_loop/sampled_mols_scores/cumulative/min"] = self.D_train.scores[:self.D_train.storage_filled].min().item()

            # Logging
            if self.iter_idx % self.config["log_interval"] == 0:
                self.log(logs)

                # df = pd.DataFrame(data=self.env.evaluate)
                # # df.to_pickle('logs/D3.pkl')
                # self.test_oracle = get_test_proxy(df)

                
                # score = self.test_oracle.give_score()
                # # wandb.log({"Performance based on classifier trained on test set": score})
                # # TODO adding to log
                # print('Iteration {}, test oracle accuracy: {}'.format(self.iter_idx, score))
                

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
            state = env.reset().clone() # Returns (46, )
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

                next_state, reward, pred_prob, done, info = env.step(action)

                if action.item() == env.EOS_idx: # Query
                    return_mols[curr_queried] = next_state
                    curr_queried += 1
                    end_traj = True 
                if done:
                    end_traj = True


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




    def log(self, logs):
        """
            Tentatively Tensorboard, but perhaps we want WANDB
        """

        # log the average weights and gradients of all models (where applicable)
        for [model, name] in [
            [self.policy, 'policy']
        ]:
            if model is not None:
                param_list = list(model.parameters())
                param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
                self.logger.add('weights/{}'.format(name), param_mean, self.iter_idx)

                if name == 'policy':
                    self.logger.add('weights/policy_std', param_list[0].data.mean(), self.iter_idx)
                
                if param_list[0].grad is not None:
                    
                    param_list_grad = [param_list[i].grad for i in range(len(param_list))]
                    param_list_grad = list(filter(lambda x : x is not None, param_list_grad))
                    
                    param_grad_mean = np.mean(
                        [param_list_grad[i].cpu().numpy().mean() for i in range(len(param_list_grad))])

                    self.logger.add('gradients/{}'.format(name), param_grad_mean, self.iter_idx)

        for k, v in logs.items():
            self.logger.add(k, v, self.iter_idx)
