
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
    "num_meta_updates_per_iter": 50, 
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
        "model_name": "RNA14_Oracle",
    },
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



rna14_002 = { # 001 but num samples per proxy set to 100
    "exp_label": "MetaRLBO-RNA14-002",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 100,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
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
        "model_name": "RNA14_Oracle",
    },
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}




rna14_003 = { # increase number of inner loop updates
    "exp_label": "MetaRLBO-RNA14-003",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 32,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
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
        "model_name": "RNA14_Oracle",
    },
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_004 = { # Increase number of meta updates
    "exp_label": "MetaRLBO-RNA14-004",
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
        "model_name": "RNA14_Oracle",
    },
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_005 = { # Lower the inner lr but increase num of inner updates
    "exp_label": "MetaRLBO-RNA14-005",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 32,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 100,
    "inner_lr": 0.5,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
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
        "model_name": "RNA14_Oracle",
    },
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_006 = { # 001 but beta 2 -> 1
    "exp_label": "MetaRLBO-RNA14-006",
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
    "num_meta_updates_per_iter": 50, 
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
            'beta': 1.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2, 
    },

    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_007 = { # 001 but beta 2 -> 0.5
    "exp_label": "MetaRLBO-RNA14-007",
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
    "num_meta_updates_per_iter": 50, 
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
            'beta': 0.5,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2, 
    },

    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

rna14_008 = { # 001 but beta 2 -> 0.0
    "exp_label": "MetaRLBO-RNA14-008",
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
    "num_meta_updates_per_iter": 50, 
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
            'beta': 0.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2, 
    },

    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}