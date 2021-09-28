import torch
from algo.diversity import hamming_distance
import numpy as np

def get_scores(config, mols, proxy_oracles, proxy_oracle_models, flatten_proxy_oracle_input, logs, iter_idx):
    if config["selection_criteria"]["method"] == "RANDOM" or iter_idx == 0:
        scores = torch.randperm(mols.shape[0]) # First iteration...

    elif config["selection_criteria"]["method"] == "PROXY_MEAN":
        proxy_scores = []
        for j in range(config["num_proxies"]):
            proxy_scores.append(torch.tensor(proxy_oracles[j].query(proxy_oracle_models[j], mols,
                                                                         flatten_input=flatten_proxy_oracle_input)))
        proxy_scores = torch.stack(proxy_scores)
        proxy_scores_mean = proxy_scores.mean(dim=0)

        scores = proxy_scores_mean

    elif config["selection_criteria"]["method"] == "UCB":
        
        if config["proxy_oracle"]["model_name"] == "GPR":
            print("SPECIAL selection for GPR")
            proxy_means = []
            proxy_stds = []
            for j in range(config["num_proxies"]):
                mean_score, std_score = proxy_oracles[j].query(proxy_oracle_models[j], mols,
                                                                             flatten_input=flatten_proxy_oracle_input,
                                                                             return_std=True)
                proxy_means.append(torch.tensor(mean_score))
                proxy_stds.append(torch.tensor(std_score))
            proxy_means = torch.stack(proxy_means)
            proxy_stds = torch.stack(proxy_stds)
            proxy_scores_mean = proxy_means.mean(dim=0)
            proxy_scores_std = proxy_stds.mean(dim = 0)

        else:
            proxy_scores = []
            for j in range(config["num_proxies"]):
                proxy_scores.append(torch.tensor(proxy_oracles[j].query(proxy_oracle_models[j], mols,
                                                                             flatten_input=flatten_proxy_oracle_input)))
            proxy_scores = torch.stack(proxy_scores)
            proxy_scores_mean = proxy_scores.mean(dim=0)

            proxy_scores_std = proxy_scores.std(dim=0)

        logs["select_molecules/proxy_model/mean/mean"] = proxy_scores_mean.mean()
        logs["select_molecules/proxy_model/std/mean"] = proxy_scores_std.mean()

        scores = proxy_scores_mean + config["selection_criteria"]["config"]["beta"] * proxy_scores_std

    else:
        raise NotImplementedError

    return logs, scores


def select(config, mols, n_query, use_diversity_metric=False):
    """
        Greedily select the molecules that are above a threshold... exponentially decrease the threshold if insufficient molecules are selected
    """

    if mols.shape[0] <= n_query:
        return mols


    if config["diversity"]["div_metric_name"] == "hamming":
        diversity_metric = hamming_distance
    else:
        raise NotImplementedError


    selected_idx = [0]
    selected_set = set(selected_idx)
    threshold = config["selection_criteria"]["diversity_threshold"]


    if not use_diversity_metric:
        return mols[:n_query]

    while len(selected_set) < n_query:
        for i in range(mols.shape[0]):
            # import pdb; pdb.set_trace()
            if i not in selected_set:
                if diversity_metric(history = mols[np.array(selected_idx)], seq = mols[i]).min() >= threshold:
                    selected_idx.append(i)
                    selected_set.add(i)
                    if len(selected_idx) == n_query:
                        return mols[np.array(selected_idx)]

        threshold = int(threshold / 2)


    assert False

    return None # This should never happen