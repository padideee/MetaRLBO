
rna14_001 = { # Based on Ising20-052
    "exp_label": "MetaRLBO-RNA14-CNN-052",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 2.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}
