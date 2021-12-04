import torch
from utils.torch_utils import weighted_mean

# def reinforce_loss(policy, episodes, params=None):
#   # Old loss from: https://github.com/tristandeleu/pytorch-maml-rl/blob/21d4ba1ccd300a403928e6db553c9529bcc3fbdf/maml_rl/utils/reinforcement_learning.py
#     pi = policy(episodes.observations.view((-1, *episodes.observation_shape)),
#                 params=params)

#     log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
#     log_probs = log_probs.view(len(episodes), episodes.batch_size)

#     losses = -weighted_mean(log_probs * episodes.advantages,
#                             lengths=episodes.lengths)

#     return losses.mean()

def reinforce_loss(episodes, params=None, use_baseline=False):
    # losses = episodes.log_probs * episodes.returns * episodes.masks
    # return -losses.sum() / episodes.masks.sum()


    if use_baseline:
    	return -weighted_mean(episodes.log_probs * episodes.advantages, dim=0, weights=episodes.masks) # Our storage is [num_steps, num_processes] not [num_processes, num_steps], so it's dim=1 instead of dim=0
    else:
    	return -weighted_mean(episodes.log_probs * episodes.returns, dim=0, weights=episodes.masks)
    
    # return -losses.sum() / episodes.num_samples
