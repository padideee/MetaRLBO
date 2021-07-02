import torch
import numpy as np
from tqdm import tqdm
import utils.helpers as utl
import utils.reinforcement_learning as rl_utl

from storage.rollout_storage import RolloutStorage
from storage.query_storage import QueryStorage
from policies.policy import Policy
from policies.random_policy import RandomPolicy


from oracles.AMP_true_oracle import AMPTrueOracle
from oracles.proxy.AMP_proxy_oracle import AMPProxyOracle

from environments.AMP_env import AMPEnv
from data.process_data import get_AMP_data
# from acquisition_functions import UCB



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MetaLearner:
    """
    Meta-Learner class with the main training loop
    """

    def __init__(self, config):
        self.config = config


        

        # D_AMP = QueryStorage(...) # TODO: Replace with https://github.com/padideee/MBRL-for-AMP/blob/main/main.py


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



        if self.config["proxy_oracle"]["model_name"] == 'RFC':
            self.flatten_proxy_oracle_input = True
        else:
            self.flatten_proxy_oracle_input = False


        # -- END ---
        
        self.env = AMPEnv(self.true_oracle_model)


        self.D_train = QueryStorage(storage_size=self.config["max_num_queries"], state_dim = self.env.observation_space.shape)

        self.proxy_oracles = [AMPProxyOracle(training_storage=self.D_train) for j in range(self.config["num_proxies"])]
        self.proxy_oracle_models = [utl.get_proxy_oracle_model(self.config) for j in range(self.config["num_proxies"])]
        self.proxy_envs = [AMPEnv(self.proxy_oracle_models[j]) for j in range(self.config["num_proxies"])]


        self.policy = None





    def meta_update(self):
        # TODO
        pass



    def run(self):

        self.true_oracle_model = self.true_oracle.fit(self.true_oracle_model, flatten_input = self.flatten_true_oracle_input)
        updated_params = [None for _ in range(self.config["num_proxies"])]

        for i in range(self.config["num_meta_updates"]):
            self.D_meta_query = RolloutStorage(num_samples = self.config["num_meta_proxy_samples"],
                                               state_dim = self.env.observation_space.shape,
                                               action_dim = 1, # Discrete value
                                               hidden_dim = self.config["policy"]["hidden_dim"],
                                               num_steps = self.env.max_AMP_length
                                               )

            logs = {} # TODO


            # Sample molecules to train proxy oracles
            if i == 0:

                random_policy = RandomPolicy(input_size = self.env.observation_space.shape, output_size = 1, num_actions=self.env.action_space.n)
                sampled_mols = self.sample_policy(random_policy, self.env, self.config["num_initial_samples"]) # Sample from true env. using random policy (num_starting_mols, dim of mol)


                sampled_mols = utl.to_one_hot(self.config, sampled_mols)
                
                sampled_mols_scores = torch.tensor(self.true_oracle.query(self.true_oracle_model, sampled_mols, flatten_input = self.flatten_true_oracle_input))
                #[Prob. False, Prob. True]


                if self.config["task"] == "AMP":
                    sampled_mols_labels = utl.scores_to_labels(self.config["true_oracle"]["model_name"], self.true_oracle_model, sampled_mols_scores)
                    # Add to storage

                    self.D_train.insert(sampled_mols, sampled_mols_labels) 
                else:
                    self.D_train.insert(sampled_mols, sampled_mols_scores) 

            else:

                for j in range(self.config["num_proxies"]):
                    sampled_mols = self.sample_policy(self.policy, self.env, self.config["num_samples_per_iter"], params=updated_params[j]) # Sample from policies -- preferably make this parallelised in the future
                    sampled_mols = utl.to_one_hot(self.config, sampled_mols)
                    sampled_mols_scores = torch.tensor(self.true_oracle.query(self.proxy_oracle_models[j], sampled_mols, flatten_input = self.flatten_true_oracle_input))


                    if self.config["task"] == "AMP":
                        sampled_mols_labels = utl.scores_to_labels(self.config["proxy_oracle"]["model_name"], self.proxy_oracle_models[j], sampled_mols_scores)

                        self.D_train.insert(sampled_mols, sampled_mols_labels)
                    else:
                        self.D_train.insert(sampled_mols, sampled_mols_scores)


            # Fit proxy oracles
            for j in range(self.config["num_proxies"]):
                self.proxy_oracle_models[j] = self.proxy_oracles[j].fit(self.proxy_oracle_models[j], flatten_input = self.flatten_proxy_oracle_input)




            # Proxy(Task)-specific updates
            for j in range(self.config["num_proxies"]):


                self.D_j = RolloutStorage(num_samples = self.config["num_samples_per_task_update"],
                                           state_dim = self.env.observation_space.shape,
                                           action_dim = 1, # Discrete value
                                           hidden_dim = self.config["policy"]["hidden_dim"],
                                           num_steps = self.env.max_AMP_length
                                           )

                # # Testing (1):
                # random_policy = RandomPolicy(input_size = self.env.observation_space.shape, output_size = 1, num_actions=self.env.action_space.n)
                # self.sample_policy(random_policy, self.proxy_envs[j], self.config["num_samples_per_task_update"], policy_storage=self.D_j) # Sample from policy[j]
                


                # Real:
                self.sample_policy(self.policy, self.proxy_envs[j], self.config["num_samples_per_task_update"], policy_storage=self.D_j) # Sample from policy[j]



                # TODO - number of inner loop updates
                loss = rl_utl.reinforce_loss(self.D_j) # Calculate loss using self.D_j - TODO: RL vs. Sup. Setting Formulation
                

                

                # # Testing (2):
                # updated_params[j] = None


                # Real:
                updated_params[j] = self.policy.update_params(loss) # Tristan's update_params for MAML-RL "https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/policies/policy.py"


            for j in range(self.config["num_proxies"]):
                # # Testing (3):
                # random_policy = RandomPolicy(input_size = self.env.observation_space.shape, output_size = 1, num_actions=self.env.action_space.n)
                # self.sample_policy(random_policy, self.proxy_envs[j], self.config["num_samples_per_task_update"], policy_storage=self.D_j) # Sample from policy[j]
                # sampled_mols = self.sample_policy(random_policy, self.proxy_envs[j], self.config["num_meta_proxy_samples"], policy_storage=self.D_meta_query, params=updated_params[j]) # Sample from policies using (update_params)


                # Real:
                sampled_mols = self.sample_policy(self.policy, self.proxy_envs[j], self.config["num_meta_proxy_samples"], policy_storage=self.D_meta_query, params=updated_params[j]) # Sample from policies using (update_params)


            # Perform meta-update
            self.meta_update()

            self.log(logs)
                

    def sample_policy(self, policy, env, num_samples, policy_storage = None, params=None):
        """
            Args:
             - Policy - 
             - env - 
             - num_samples - 
             - policy_storage: Optional -- Used to store the rollout for training the pollicy
             - params: Optional -- Use these parameters instead of the default parameters for the policy (MAML)

            Return:
             - Molecules: Tensor of size (num_samples, dim of mol)

        """
        # import pdb; pdb.set_trace()
        state_dim = env.observation_space.shape # Currently hardcoded
        return_mols = torch.zeros(num_samples, *state_dim)

        for j in range(num_samples):


            done = False
            state = env.reset()
            while not done:

                if params is not None:
                    action, log_prob = policy(state, params)
                else:
                    action, log_prob = policy(state)

                next_state, reward, pred_prob, done, info = env.step(action)

                done = torch.tensor(done)
                if done.item():
                    return_mols[j] = next_state


                if policy_storage is not None:
                    policy_storage.insert(state=state, 
                                   next_state=next_state,
                                   action=action, 
                                   reward=reward,
                                   log_prob=log_prob,
                                   done=done)

                state = next_state

            if policy_storage is not None:
                policy_storage.compute_returns()
                policy_storage.after_rollout()


        return return_mols






    def log(self, logs):
        """
            TODO

        """


        return
