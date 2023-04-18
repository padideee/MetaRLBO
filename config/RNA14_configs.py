
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



rna14_009 = { # 006 but (num_samples_per_proxy: 64 -> 32) -- should see a noticeable speedup... 
    "exp_label": "MetaRLBO-RNA14-009",
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


rna14_010 = { # 008 but reset policy per round
    "exp_label": "MetaRLBO-RNA14-010",
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_011 = { # 010 but 50 -> 100 num_meta_updates_per_iter
    "exp_label": "MetaRLBO-RNA14-011",
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
    "num_meta_updates_per_iter": 100, 
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



rna14_012 = { # 010 but num_steps: 150 -> 58, num_meta_steps: 50 -> 58 (should see a 3x speed up or so) -- what about the results that we'll see though...
    "exp_label": "MetaRLBO-RNA14-012",
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
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_013 = { # 012 but num_inner_updates: 1 -> 2
    "exp_label": "MetaRLBO-RNA14-013",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
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
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_014 = { # 011 (but il, ol: 1.0, 0.1 -> 0.5, 0.05)
    "exp_label": "MetaRLBO-RNA14-014",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 100,
    "inner_lr": 0.5,
    "outer_lr": 0.05,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



rna14_015 = { # 011 (but il, ol: 1.0, 0.1 -> 0.5, 0.05 and num_meta_updates: 50 -> 80)
    "exp_label": "MetaRLBO-RNA14-015",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 100,
    "inner_lr": 0.5,
    "outer_lr": 0.05,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



rna14_016 = { # 011 (but il, ol: 1.0, 0.1 -> 0.5, 0.05 and radius: 2 -> 1)
    "exp_label": "MetaRLBO-RNA14-016",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 100,
    "inner_lr": 0.5,
    "outer_lr": 0.05,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_017 = { # 011 (but il, ol: 1.0, 0.1 -> 0.5, 0.05 and radius: 2 -> 1, reset_policy: True -> False)
    "exp_label": "MetaRLBO-RNA14-017",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 100,
    "inner_lr": 0.5,
    "outer_lr": 0.05,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



rna14_018 = { # 011 (but il, ol: 1.0, 0.1 -> 0.1, 0.01 and radius: 2 -> 1, reset_policy: True -> False)
    "exp_label": "MetaRLBO-RNA14-018",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 100,
    "inner_lr": 0.1,
    "outer_lr": 0.01,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



rna14_019 = { # 011 (but il, ol: 1.0, 0.1 -> 0.1, 0.01 and radius: 2 -> 1, reset_policy: True -> False, and num_meta_updates: 50 -> 100)
    "exp_label": "MetaRLBO-RNA14-019",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 100,
    "inner_lr": 0.1,
    "outer_lr": 0.01,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}




rna14_020 = { # 013 but num_meta_updates_per_iter: 50 -> 80
    "exp_label": "MetaRLBO-RNA14-020",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
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
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

rna14_020_MLP = { # 020 but with MLP instead of CNN
    "exp_label": "MetaRLBO-RNA14-020_MLP",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
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
        "model_name": "MLP",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

rna14_020_KNR = { # 020 but with KNR instead of CNN
    "exp_label": "MetaRLBO-RNA14-020_KNR",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
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
        "model_name": "KNR",
        "p": 0.8, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

rna14_020_BR = { # 020 but with KNR instead of CNN
    "exp_label": "MetaRLBO-RNA14-020_BR",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
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
        "model_name": "BR",
        "p": 0.8, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

rna14_020_RR = { # 020 but with KNR instead of CNN
    "exp_label": "MetaRLBO-RNA14-020_RR",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
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
        "model_name": "RR",
        "p": 0.8, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_021 = { # 013 but lambda: 0.1 -> 0.2
    "exp_label": "MetaRLBO-RNA14-021",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
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
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
        "lambda": 0.2, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2, 
    },

    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_022 = { # 013 but num_meta_updates_per_iter: 50 -> 100
    "exp_label": "MetaRLBO-RNA14-022",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 32,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 100, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_023 = { # 020 but radius: 2 -> 3
    "exp_label": "MetaRLBO-RNA14-023",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
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
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_024 = { # 020 but num_inner_updates: 2 -> 3
    "exp_label": "MetaRLBO-RNA14-024",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
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
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



rna14_025 = { # 023 but beta: 1 -> 2
    "exp_label": "MetaRLBO-RNA14-025",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
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
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



rna14_026 = { # 023 but beta: 1 -> 0
    "exp_label": "MetaRLBO-RNA14-026",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
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
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

rna14_026_qp_1 = { # 023 but beta: 1 -> 0
    "exp_label": "MetaRLBO-RNA14-026-QP-1",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 1,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 2048,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_026_qp_2 = { # 023 but beta: 1 -> 0
    "exp_label": "MetaRLBO-RNA14-026-QP-2",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 2,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 1024,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

rna14_026_qp_4 = { # 023 but beta: 1 -> 0
    "exp_label": "MetaRLBO-RNA14-026-QP-4",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 4,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 512,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_026_qp_8 = { # 023 but beta: 1 -> 0
    "exp_label": "MetaRLBO-RNA14-026-QP-8",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

rna14_026_qp_16 = { # 023 but beta: 1 -> 0
    "exp_label": "MetaRLBO-RNA14-026-QP-16",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 16,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 128,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, 
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


rna14_027 = { # 020 but num_meta_updates_per_iter: 50 -> 80
    "exp_label": "MetaRLBO-RNA14-027",
    "task": "RNA14-v0",
    "num_proxies": 4, 
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
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
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


##### Thesis exp

Thesis_rna14_026_qp_8 = { # 023 but beta: 1 -> 0
    "exp_label": "Thesis_MetaRLBO-RNA14-026-QP-8",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


Thesis_rna14_027_qp_8 = { # 023 but beta: 1 -> 0
    "exp_label": "Thesis_MetaRLBO-RNA14-027-QP-8",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "PROXY_MEAN",
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


Thesis_rna14_028_qp_8 = { # 023 but beta: 1 -> 0
    "exp_label": "Thesis_MetaRLBO-RNA14-028-QP-8",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "PI",
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

# --------------------------------- RND
Thesis_rna14_026_RND_1 = { # 023 but beta: 1 -> 0
    "exp_label": "Thesis_MetaRLBO-RNA14-026-RND_1",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
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
        "lambda": - 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },

    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

########## test
Thesis_rna14_026_RND_beta0 = { # RND on, E + IN, beta=0
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_beta0",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "E+IN",
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
        "lambda": - 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },

    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_IN = { # only IN reward (with RND)
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_IN",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "IN",
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
        "lambda": - 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },

    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_beta1 = { # IN+E, MCdropout, beta=1.0, lambda=0.1
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_beta1",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "E+IN",
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB",
        "config": {
            'beta': 1.0, #0.0
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": - 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },

    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_beta1_bootstrap = { # IN+E, MCdropout+ensemble, beta=1.0
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_beta1_bootstrap",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, #1.0
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "E+IN",
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB",
        "config": {
            'beta': 1.0, #0.0
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": - 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },

    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_ham_beta1_bootstrap = { # IN+E, MCdropout+ensemble, beta=1.0, RND -> hamming
    "exp_label": "Thesis_MetaRLBO-RNA14-026_ham_beta1_bootstrap",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, #1.0
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "hamming", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "E+IN",
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB",
        "config": {
            'beta': 1.0, #0.0
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_ham_beta0_bootstrap = { # IN+E, MCdropout+ensemble, beta=1.0 -> beta=0.0, hamming
    "exp_label": "Thesis_MetaRLBO-RNA14-026_ham_beta0_bootstrap",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, #1.0
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "hamming", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "E+IN",
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB",
        "config": {
            'beta': 0.0, #1.0
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

############
Thesis_rna14_026_RND_beta0_lambda1 = { # RND on, E + IN, beta=0, lambda: 0.1 -> 1.0
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_beta0_lambda1",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "E+IN",
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
        "lambda": - 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_s_beta1_lambda1 = { # RND on, E + IN, beta=1, lambda: 0.1 -> 1.0
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_s_beta1_lambda1",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON", # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
            "RND_metric": "soft",
			"T": 1,
    },
    "reward" : "E+IN",
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
        "lambda": - 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_c_beta1_lambda1 = { # RND on, E + IN, beta=1, lambda: 0.1 -> 1.0, RND: s -> c
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_c_beta1_lambda1",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON", # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
            "RND_metric": "cosine",
			"T": 1,
    },
    "reward" : "E+IN",
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
        "lambda": - 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_s_beta1_lambda005 = { # RND on, E + IN, beta=0, lambda:  0.1, RND: s
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_s_beta1_lambda005",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON", # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
            "RND_metric": "soft",
			"T": 1,
    },
    "reward" : "E+IN",
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
        "lambda": - 0.05, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_c_beta1_lambda005 = { # RND on, E + IN, beta=0, lambda:  0.05, RND: c
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_c_beta1_lambda005",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON", # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
            "RND_metric": "cosine",
			"T": 1,
    },
    "reward" : "E+IN",
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
        "lambda": - 0.05, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_c_beta1_lambda2 = { # RND on, E + IN, beta=0, lambda: 2, RND: c
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_c_beta1_lambda2",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON", # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
            "RND_metric": "cosine",
			"T": 1,
    },
    "reward" : "E+IN",
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
        "lambda": - 2, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_s_beta1_lambda2 = { # RND on, E + IN, beta=0, lambda: 2, RND: s
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_s_beta1_lambda2",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON", # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
            "RND_metric": "soft",
			"T": 1,
    },
    "reward" : "E+IN",
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
        "lambda": - 2, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

#### bests
Thesis_rna14_026_ham_beta1_lambda01_NoPE = { # No Positional encoding
    "exp_label": "Thesis_MetaRLBO-RNA14-026_ham_beta1_lambda01_NoPE",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "hamming", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "E+IN",
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB",
        "config": {
            'beta': 1.0, #0.0
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_ham_beta1_lambda01_PE = { # with Positional encoding
    "exp_label": "Thesis_MetaRLBO-RNA14-026_ham_beta1_lambda01_PE",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "hamming", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "E+IN",
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB",
        "config": {
            'beta': 1.0, #0.0
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
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

###### Annealing
Thesis_rna14_026_ham_beta1_anneal_r7 = { # radius:7, annealing
    "exp_label": "Thesis_MetaRLBO-RNA14-026_ham_beta1_anneal_r7",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "hamming", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "E+IN",
    "reward_annealing": True,
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB",
        "config": {
            'beta': 1.0, #0.0
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 10, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 7,
    },

    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_ham_beta1_NOanneal_r7 = { # radius:7, NOT annealing
    "exp_label": "Thesis_MetaRLBO-RNA14-026_ham_beta1_NOanneal_r7",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "hamming", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "E+IN",
    "reward_annealing": False,
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB",
        "config": {
            'beta': 1.0, #0.0
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 7,
    },

    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_ham_beta1_NOanneal_r7_lambda_01 = { # radius:7, NOT annealing
    "exp_label": "Thesis_MetaRLBO-RNA14-026_ham_beta1_NOanneal_r7_lambda_01",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "hamming", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
    "reward" : "E+IN",
    "reward_annealing": False,
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB",
        "config": {
            'beta': 1.0, #0.0
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 7,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



Thesis_rna14_026_RND_s_beta1_anneal = { # anneal, RND
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_s_beta1_anneal",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON", # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
            "RND_metric": "soft",
			"T": 1,
    },
    "reward" : "E+IN",
    "reward_annealing": True,
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
        "lambda": - 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_c_beta1_anneal = { # anneal, RND
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_c_beta1_anneal",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON", # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
            "RND_metric": "cosine",
			"T": 1,
    },
    "reward" : "E+IN",
    "reward_annealing": True,
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
        "lambda": - 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_c_beta1_anneal_p = { # anneal, RND, reward to penalty
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_c_beta1_anneal_p",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "reward_transofrm": "to_penalty",
            "div_switch": "ON", # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
            "RND_metric": "cosine",
			"T": 1,
    },
    "reward" : "E+IN",
    "reward_annealing": True,
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
        "lambda":  10, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

Thesis_rna14_026_RND_c_beta1_NOanneal_p = { # anneal, RND, reward to penalty
    "exp_label": "Thesis_MetaRLBO-RNA14-026_RND_c_beta1_NOanneal_p",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 8,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 100,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0,
    },
    "policy": {
        "num_steps": 58, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
        "num_meta_steps": 58,
    },
    "exp_policy": {
            "model_name": "MLP",
            "input_size": 56,  # 14*4
            "hidden_size": 28,  # 14 * 2
            "output_size": 14,
    },
    "diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "reward_transofrm": "to_penalty",
            "div_switch": "ON", # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
            "RND_metric": "cosine",
			"T": 1,
    },
    "reward" : "E+IN",
    "reward_annealing": False,
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
        "lambda":  1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2,
    },
    "true_oracle": {
        "model_name": "RNA14_Oracle",
    },
    "reset_policy_per_round": True,
    "use_baseline": False,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}