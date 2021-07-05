import torch
import numpy as np
from torch import optim
from tqdm import tqdm
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
        D_AMP = get_AMP_data('data/data.hkl') 

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
        
        self.env = AMPEnv(self.true_oracle_model)

        self.D_train = QueryStorage(storage_size=self.config["max_num_queries"], state_dim = self.env.observation_space.shape)

        self.proxy_oracles = [AMPProxyOracle(training_storage=self.D_train, p=self.config["proxy_oracle"]["p"]) for j in range(self.config["num_proxies"])]
        self.proxy_oracle_models = [utl.get_proxy_oracle_model(self.config) for j in range(self.config["num_proxies"])]
        self.proxy_envs = [AMPEnv(self.proxy_oracle_models[j]) for j in range(self.config["num_proxies"])]


        self.policy = CategoricalGRUPolicy(num_actions = self.env.action_space.n + 1,
                                            hidden_size = self.config["policy"]["hidden_dim"],
                                            state_dim = self.env.action_space.n + 1,
                                            state_embed_dim = self.config["policy"]["state_embedding_size"],
                                            )


        self.iter_idx = 0


    def run(self):

        self.true_oracle_model = self.true_oracle.fit(self.true_oracle_model, flatten_input = self.flatten_true_oracle_input)
        updated_params = [None for _ in range(self.config["num_proxies"])]

        for self.iter_idx in tqdm(range(self.config["num_meta_updates"])):
            self.D_meta_query = RolloutStorage(num_samples = self.config["num_meta_proxy_samples"],
                                               state_dim = self.env.observation_space.shape,
                                               action_dim = 1, # Discrete value
                                               hidden_dim = self.config["policy"]["hidden_dim"],
                                               num_steps = self.env.max_AMP_length
                                               )

            logs = {} 


            # Sample molecules to train proxy oracles
            if self.iter_idx == 0:

                random_policy = RandomPolicy(input_size = self.env.observation_space.shape, output_size = 1, num_actions=self.env.action_space.n)
                sampled_mols = self.sample_policy(random_policy, self.env, self.config["num_initial_samples"]) # Sample from true env. using random policy (num_starting_mols, dim of mol)


                sampled_mols = utl.to_one_hot(self.config, sampled_mols) # Leo: There could be an issue with the end of state token...
                
                sampled_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, sampled_mols, flatten_input = self.flatten_true_oracle_input))
                #[Prob. False, Prob. True]

                self.D_train.insert(sampled_mols, sampled_mols_scores[:, 1]) 

                print(sampled_mols_scores[:, 1])

            # Fit proxy oracles
            for j in range(self.config["num_proxies"]):
                self.proxy_oracle_models[j] = self.proxy_oracles[j].fit(self.proxy_oracle_models[j], flatten_input = self.flatten_proxy_oracle_input)


            inner_opt = optim.SGD(self.policy.parameters(), lr=1e-1)
            meta_opt = optim.SGD(self.policy.parameters(), lr=1e-3)
            meta_opt.zero_grad()

            # Proxy(Task)-specific updates
            for j in range(self.config["num_proxies"]):


                with higher.innerloop_ctx(
                        self.policy, inner_opt, copy_initial_weights=False
                ) as (inner_policy, diffopt):
                    self.D_j = RolloutStorage(num_samples = self.config["num_samples_per_task_update"],
                                               state_dim = self.env.observation_space.shape,
                                               action_dim = 1, # Discrete value
                                               hidden_dim = self.config["policy"]["hidden_dim"],
                                               num_steps = self.env.max_AMP_length
                                               )
                    


                    self.sample_policy(self.policy, self.proxy_envs[j], self.config["num_samples_per_task_update"], policy_storage=self.D_j) # Sample from policy[j]



                    for k in range(self.config["num_inner_updates"]):
                        
                        inner_loss = rl_utl.reinforce_loss(self.D_j) # Leo: This is bugged -- the log_probs need to be recalculated
                        
                        logs["inner_loop/proxy-{j}/loss/{k}"] = inner_loss.item()
                        # Inner update
                        diffopt.step(inner_loss)# Need to make this differentiable... not immediately differentiable from the storage!



                    # Sample mols for meta update
                    sampled_mols = self.sample_policy(inner_policy, self.proxy_envs[j], self.config["num_meta_proxy_samples"], policy_storage=self.D_meta_query) # Sample from policies using (update_params)




                    # Sample mols for training the proxy oracle later
                    sampled_mols = self.sample_policy(inner_policy, self.env, self.config["num_samples_per_iter"]) # Sample from policies -- preferably make this parallelised in the future
                    sampled_mols = utl.to_one_hot(self.config, sampled_mols)
                    sampled_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, sampled_mols, flatten_input = self.flatten_true_oracle_input))

                    logs["inner_loop/proxy-{j}/sampled_mols_scores_avg"] = sampled_mols_scores.mean().item()


                    self.D_train.insert(sampled_mols, sampled_mols_scores[:, 1])

            outer_loss = rl_utl.reinforce_loss(self.D_meta_query) # Need to make this differentiable... not differentiable from the storage!
            print("Outer Loss:", outer_loss)
            logs["outer_loss"] = outer_loss.item()
            outer_loss.backward() # Not really...
            meta_opt.step()


            # Logging
            if self.iter_idx % self.config["log_interval"] == 0:
                self.log(logs)
                

    def sample_policy(self, policy, env, num_samples, policy_storage = None):
        """
            Args:
             - policy 
             - env 
             - num_samples - number of molecules to return
             - policy_storage: Optional -- Used to store the rollout for training the pollicy

            Return:
             - Molecules: Tensor of size (num_samples, dim of mol)

        """
        state_dim = env.observation_space.shape # Currently hardcoded
        return_mols = torch.zeros(num_samples, *state_dim)

        for j in range(num_samples):


            done = False
            state = env.reset() # Returns (46, )
            hidden_state = None
            curr_timestep = 0
            while not done:
                
                onehot_state = utl.to_one_hot(self.config, state)

                if curr_timestep == 0:
                    s = torch.zeros((1, 1, 21))
                else:
                    s = onehot_state[curr_timestep - 1].unsqueeze(0).unsqueeze(0)
                
                action, log_prob, hidden_state = policy(s, hidden_state)


                next_state, reward, pred_prob, done, info = env.step(action)

                done = torch.tensor(done)

                if done.item():
                    return_mols[j] = next_state


                if policy_storage is not None:
                    policy_storage.insert(state=state, 
                                   next_state=next_state,
                                   action=torch.tensor(action), 
                                   reward=reward,
                                   log_prob=log_prob,
                                   done=done)

                state = next_state
                curr_timestep += 1

            if policy_storage is not None:
                policy_storage.compute_returns()
                policy_storage.after_rollout()


        return return_mols




    def log(self, logs):
        """
            Tentatively Tensorboard, but perhaps we want WANDB
        """


        for k, v in logs.items():
            self.logger.add(k, v, self.iter_idx)
