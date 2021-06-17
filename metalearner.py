import torch
import numpy as np
from tqdm import tqdm
import utils.helpers as utl

from models.online_storage import RolloutStorage
from models.query_storage import QueryStorage
from models.policy import Policy
from models.random_policy import RandomPolicy


from oracles.AMP_true_oracle import AMPTrueOracle
from oracles.proxy.AMP_proxy_oracle import AMPProxyOracle

from environments.AMP_env import AMPEnv

# from acquisition_functions import UCB



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MetaLearner:
    """
    Meta-Learner class with the main training loop
    """

    def __init__(self, config):
        self.config = config


        

        # D_AMP = QueryStorage(...) # TODO: Replace with https://github.com/padideee/MBRL-for-AMP/blob/main/main.py
        D_AMP = None # TODO
        self.true_oracle = AMPTrueOracle(training_storage=D_AMP)
        self.true_oracle_model = utl.get_true_oracle_model(self.config)
        self.env = AMPEnv(self.true_oracle)

        self.proxy_oracles = [AMPProxyOracle(training_storage=self.D_train) for j in range(self.config["num_proxies"])]
        self.proxy_oracle_models = [utl.get_proxy_oracle_model(self.config) for j in range(self.config["num_proxies"])]
        self.proxy_envs = [AMPEnv(self.proxy_oracles[j]) for j in range(self.config["num_proxies"])]


        # self.policy = NormalMLPPolicy(...) # 
        self.policy = None
        self.D_train = QueryStorage(storage_size=self.config["max_num_queries"], state_dim = self.env.observation_space.shape)





    def meta_update(self):
        pass



    def run(self):

        """
            TODO:

             - Loss Calculation
             - Meta-update

        """

        self.true_oracle_model = self.true_oracle.fit(self.true_oracle_model)
        updated_params = [None for _ in range(self.config["num_proxies"])]

        for i in range(self.config["num_meta_updates"]):
            self.D_meta_query = RolloutStorage(num_samples = self.config["num_meta_proxy_samples"],
                                               state_dim = self.env.observation_space.shape,
                                               action_dim = 1, # Discrete value
                                               hidden_dim = self.config["policy"]["hidden_dim"],
                                               num_steps = self.env.max_AMP_length
                                               )

            logs = {}


            # Sample molecules to train proxy oracles
            if i == 0:

                random_policy = RandomPolicy(input_size = self.env.observation_space.shape, output_size = 1, num_actions=self.env.action_space.n)
                sampled_mols = self.sample_policy(random_policy, self.env, self.config["num_initial_samples"]) # Sample from true env. using random policy (num_starting_mols, dim of mol)

                sampled_mols_scores = true_oracle.query(self.true_oracle_model, sampled_mols)

                # Add to storage

                self.D_train.insert(sampled_mols, sampled_mols_scores)

            else:

                for j in range(self.config["num_proxies"]):
                    sampled_mols = self.sample_policy(self.policy, self.env, self.config["num_samples_per_iter"], params=updated_params) # Sample from policies -- preferably make this parallelised in the future
                    sampled_mols_scores = true_oracle.query(self.proxy_oracle_models[j], sampled_mols)


                    self.D_train.insert(sampled_mols, sampled_mols_scores)


            # Fit proxy oracles
            for j in range(self.config["num_proxies"]):
                self.proxy_oracle_models[j] = self.proxy_oracles[j].fit(self.proxy_oracle_models[j])




            # Proxy(Task)-specific updates
            for j in range(self.config["num_proxies"]):


                self.D_j = RolloutStorage(num_samples = self.config["num_samples_per_task_update"],
                                           state_dim = self.env.observation_space.shape,
                                           action_dim = 1, # Discrete value
                                           hidden_dim = self.config["policy"]["hidden_dim"],
                                           num_steps = self.env.max_AMP_length
                                           )

                self.sample_policy(self.policy, self.proxy_envs[j], self.config["num_samples_per_task_update"], storage=self.D_j) # Sample from policy[j]

                # sampled_mols_scores = self.proxy_oracles[j].query(self.proxy_oracle_models[j], sampled_mols)


                # self.D_j.insert(sampled_mols, sampled_mols_scores) # TODO: We need to replace this with reward...

                loss = ... # Calculate loss using self.D_j
                updated_params[j] = self.policy.update_params(loss) # Tristan's update_params for MAML-RL "https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/policies/policy.py"


            for j in range(self.config["num_proxies"]):

                sampled_mols = self.sample_policy(self.policy, self.proxy_envs[j], self.config["num_meta_proxy_samples"], storage=self.D_meta_query, params=updated_params) # Sample from policies using (update_params)

                sampled_mols_scores = self.proxy_oracles[j].query(self.proxy_oracle_models[j], sampled_mols)

                self.D_meta_query.insert(sampled_mols, sampled_mols_scores)



            # Perform meta-update
            self.meta_update()

            self.log(logs)
                

    def sample_policy(policy, env, num_samples, policy_storage = None, params=None):
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
        state_dim = env.observation_space.shape # Currently hardcoded
        return_mols = torch.zeros(num_samples, state_dim)

        for j in range(num_samples):


            done = False
            state = env.reset()
            while not done:

                if params is not None:
                    action = policy(state, params)
                else:
                    action = policy(state)

                next_state, reward, pred_prob, done, info = env.step(action)

                if done:
                    return_mols[j] = next_state


                if policy_storage is not None:
                    policy_storage.insert(state=state, 
                                   next_state=next_state,
                                   action=action, 
                                   reward=reward,
                                   pred_prob=pred_prob,
                                   done=done)

                state = next_state

            policy_storage.after_rollout()

        return return_mols


    def log(self, logs):
        """
            

        """


        return
