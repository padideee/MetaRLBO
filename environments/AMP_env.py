import gym
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from data.process_data import seq_to_encoding
from algo.diversity import diversity 
import torch.nn.functional as F


class AMPEnv(gym.Env):
    def __init__(self, reward_oracle, lambd = 0.1, radius = 2, max_AMP_length = 51, query_history = None):


        # Actions in AMP design are the 20 amino acids
        # For non-finite horizon case: An extra action is added to
        # represent the "end of sequence" token
        self.max_AMP_length = max_AMP_length
        self.num_actions = 21
        self.EOS_idx = 20

        self.action_space = gym.spaces.Discrete(self.num_actions) # 20 amino acids, End of Sequence Token

        # The state at time t is given by the last t tokens in the AMP sequence
        # TODO: Change observation size to window W
        # We limit the sequence to 50 characters of AMP, and an extra character for EOS
        self.obs_shape = [max_AMP_length, self.num_actions]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=float)

        self.start_state = torch.tensor(np.zeros(self.obs_shape)) 
        self.curr_state = self.start_state  # TODO: Need to update this outside of class

        self.time_step = 0 # TODO: Need to update this outside of class
        self.history = query_history if query_history is not None else []
        self.evaluate = {'seq': [], 'embed_seq': [], 'reward': [], 'pred_prob': []}

        self.max_AMP_length = max_AMP_length
        self.reward_oracle = reward_oracle
        self.proxy_oracles = []
        self.modelbased = False

        self.lambd = lambd  # TODO: tune + add this to config
        self.radius = radius

    def update_proxy_oracles(self, oracle):
        self.proxy_oracles = oracle

    def update_opt_method(self, modelbased):
        self.modelbased = modelbased

    def step(self, action):
        # Return: (state, reward, done, info)
        # NOTE: Reward is the prediction probability of whether
        # a sequence is antimicrobial towards a certain pathogen
        done = False
        reward = torch.tensor(0.0)
        pred_prob = torch.tensor([0.0, 0.0])

        
        self.curr_state[self.time_step] = F.one_hot(action, num_classes = self.num_actions)

        queried = False
        if action.item() == self.EOS_idx:
            queried = True
            done = True
            # compute density of similar sequences in the history
            if len(self.history) > 1:
                # Use div=False to test without diversity promotion
                dens = diversity(self.curr_state, self.history, div=True, radius = self.radius).density()
            else:
                dens = 0.0
            # self.history.append(self.curr_state)

            # Store predictive probability for regression
            if self.modelbased:
                # NOTE: This is regression case -> the oracle predicts the probability of given
                # sequence to be AMP-like i.e. prob. of being antimicrobial towards a certain pathogen
                # print("Model based: ", predict_seq)

                pred = []
                for m in self.proxy_oracles:
                    # s = seq_to_encoding(self.curr_state.unsqueeze(0)) # seq_to_encoding -- not the one hot encoding...
                    s = self.curr_state.unsqueeze(0)
                    d = m.predict(s.numpy()[np.newaxis, :])
                    pred.append(d)
                predictionAMP = np.average(pred)
                # print("Avg. prediction: ", predictionAMP)
                # Return avg. prediction based on proxy models
                pred_prob = torch.tensor([[1 - predictionAMP, predictionAMP]])
                reward = torch.tensor(predictionAMP)
                reward -= self.lambd * dens

                with open('logs/log.txt', 'a+') as f:
                    f.write('Model Based' + '\t' + str(reward.detach().cpu().numpy()) + '\n')
                self.evaluate['seq'].append(self.curr_state.detach().cpu().numpy())
                self.evaluate['embed_seq'].append(s.detach().cpu().numpy())
                self.evaluate['reward'].append(reward.detach().cpu().numpy())
                self.evaluate['pred_prob'].append(predictionAMP)

                # wandb.log({"train_pred_prob": predictionAMP})

            else:
                # (returns prob. per classification class --> [Prob. Neg., Prob. Pos.])

                # s = seq_to_encoding(self.curr_state.unsqueeze(0)) # Leo: TODO (this takes as input -- not the one hot encoding...)
                s = self.curr_state.unsqueeze(0).flatten(-2, -1)
                try:
                    pred_prob = torch.tensor(self.reward_oracle.predict_proba(s))
                    reward = pred_prob[0][1] 
                except:
                    reward = torch.tensor(self.reward_oracle.predict(s))
                    pred_prob = torch.tensor([[1 - reward, reward]])
                # # ---- Modification ---------
                # """
                #     Modification to env.:

                #     There's a special case where if the reward oracle was only trained on neg./pos. data, then it will output a
                #     pred_prob w/ a final dimension of 1, which leads to an error.
                # """
                # assert self.reward_oracle.classes_.shape[-1] <= 2

                # if pred_prob.shape[-1] == 1:
                    
                #     prob = torch.zeros((*pred_prob.shape[:-1], 2))

                #     if self.reward_oracle.classes_[0] == 0:
                #         prob[:, 0] = 1
                #     elif self.reward_oracle.classes_[0] == 1:
                #         prob[:, 1] = 1

                #     pred_prob = prob

                # # ---- End Modification -----

                   
                with open('logs/log.txt', 'a+') as f:
                    f.write('Model Free' + '\t' + str(reward.detach().cpu().numpy()) + '\n')
                self.evaluate['seq'].append(self.curr_state.detach().cpu().numpy())
                self.evaluate['embed_seq'].append(s.detach().cpu().numpy())
                self.evaluate['reward'].append(reward.detach().cpu().numpy())
                self.evaluate['pred_prob'].append(pred_prob[0][1].detach().cpu().numpy())

                # wandb.log({"train_pred_prob": pred_prob[0][1].detach().cpu().numpy()})

                #TODO adding the pred_prob to Tensorboard log


        self.time_step += 1

        if self.time_step >= self.max_AMP_length:
            done = True

        # Info must be a dictionary
        info = [{"action": action, "state": self.curr_state, "pred_prob": pred_prob, "queried": queried}]

        return(self.curr_state, reward, pred_prob, done, info)

    def reset(self):
        self.curr_state = torch.tensor(np.zeros(self.obs_shape))
        self.time_step = 0
        return self.curr_state

"""
def main():
    env = AMPEnv()
    print(check_env(env))
if __name__ == "__main__":
    main()
"""



