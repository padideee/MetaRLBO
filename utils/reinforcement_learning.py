import torch

# def reinforce_loss(policy, episodes, params=None):
#   # Old loss from: https://github.com/tristandeleu/pytorch-maml-rl/blob/21d4ba1ccd300a403928e6db553c9529bcc3fbdf/maml_rl/utils/reinforcement_learning.py
#     pi = policy(episodes.observations.view((-1, *episodes.observation_shape)),
#                 params=params)

#     log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
#     log_probs = log_probs.view(len(episodes), episodes.batch_size)

#     losses = -weighted_mean(log_probs * episodes.advantages,
#                             lengths=episodes.lengths)

#     return losses.mean()

def reinforce_loss(episodes, params=None):
    losses = episodes.log_probs * episodes.returns * episodes.masks

    return -losses.sum() / episodes.masks.sum()
    # return -losses.sum() / episodes.num_samples

def entropy_bonus(episodes):
	return - (episodes.log_probs).sum()
