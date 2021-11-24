
dynappo_ising50_001 = {
    "exp_label": "dynappo_ising50_001",
    "task": "AltIsing50-v0",
    "use_metalearner": False,
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "query_storage_size": 100000, # Maximum number of queries allowed in the storage
    "num_updates_per_iter": 72,
    "num_initial_samples": 500,
    "num_query_per_iter": 500,
    "num_samples_per_iter": 800,
    "ppo_config": { # Leo: This should be merged into train_policy_config --
        "clip_param": 0.2,
        "ppo_epoch": 4,
        "num_mini_batch": 4,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "lr": 7e-4,
        "eps": 1e-5,
        "max_grad_norm": 0.5,
        "use_gae": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "use_proper_time_limits": False,
        # "num_steps": 10,
    },
    "proxy_oracle": {
        "p": 1.0, # Proportion of data to sample to train proxy oracles -- Fixed to 1.0 for DynaPPO!
    },
    "policy": {
        "num_steps": 200,
    },  
    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "save_interval": 10, # Save model every n batch queries
    "num_processes": 8,
    "results_log_dir": "./logs", 
    "mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
    "seed": 73,
}


############## MetaRLBO


small_metarlbo_ising50_001 = { 
    "exp_label": "Small-MetaRLBO-Ising50-KNR-001",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 5000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 4,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 16,
    "num_query_per_iter": 10,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 1, 
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.7, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

small_metarlbo_ising50_002 = { 
    "exp_label": "Small-MetaRLBO-Ising50-KNR-002",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 5000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 4,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 16,
    "num_query_per_iter": 10,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 1, 
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.7, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

small_metarlbo_ising50_003 = { 
    "exp_label": "Small-MetaRLBO-Ising50-KNR-003",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 5000, # Maximum number of queries in experiment
    "num_inner_updates": 3,
    "num_query_proxies": 4,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 16,
    "num_query_per_iter": 10,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 1, 
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.7, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}





# Large Batches - Size 500



metarlbo_ising50_001 = { 
    "exp_label": "MetaRLBO-Ising50-KNR-001",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 16,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.7, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising50_002 = { 
    "exp_label": "MetaRLBO-Ising50-KNR-002",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_initial_samples": 500,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising50_003 = { 
    "exp_label": "MetaRLBO-Ising50-KNR-003",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 30,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising50_004 = { 
    "exp_label": "MetaRLBO-Ising50-KNR-004",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 30,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising50_005 = { 
    "exp_label": "MetaRLBO-Ising50-KNR-005",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 30,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



metarlbo_ising50_006 = {  # Test entropy reg. More entropy reg vs 001
    "exp_label": "MetaRLBO-Ising50-KNR-006",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 16,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 1.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.7, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising50_007 = {  # Test entropy reg. More entropy reg vs 001
    "exp_label": "MetaRLBO-Ising50-KNR-007",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 16,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 2.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.7, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising50_008 = { 
    "exp_label": "MetaRLBO-Ising50-KNR-008",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 30,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2, 
    },


    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising50_009 = { 
    "exp_label": "MetaRLBO-Ising50-KNR-009",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 150,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 30,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "KNR",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 1, 
    },


    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising50_010 = {  # MLP Experiment...
    "exp_label": "MetaRLBO-Ising50-MLP-010",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising50_011 = {  # MLP Experiment... (more proxies compared to 010)
    "exp_label": "MetaRLBO-Ising50-MLP-011",
    "task": "AltIsing50-v0",
    "num_proxies": 8, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising50_012 = {  # MLP Experiment... More inner loop updates compared to 001
    "exp_label": "MetaRLBO-Ising50-MLP-012",
    "task": "AltIsing50-v0",
    "num_proxies": 8, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising50_013 = {  # MLP Experiment... More inner loop updates compared to 001
    "exp_label": "MetaRLBO-Ising50-MLP-013",
    "task": "AltIsing50-v0",
    "num_proxies": 8, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 5.0,
    "outer_lr": 0.5,
    "num_meta_updates_per_iter": 50,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising50_014 = {  # MLP Experiment... More inner loop updates compared to 001
    "exp_label": "MetaRLBO-Ising50-MLP-014",
    "task": "AltIsing50-v0",
    "num_proxies": 8, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 5.0,
    "outer_lr": 0.5,
    "num_meta_updates_per_iter": 30,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising50_015 = {  # MLP Experiment... More inner loop updates compared to 001
    "exp_label": "MetaRLBO-Ising50-MLP-015",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 5.0,
    "outer_lr": 0.5,
    "num_meta_updates_per_iter": 30,
    "entropy_reg_coeff": 1.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising50_016 = {  # MLP Experiment... More inner loop updates compared to 001
    "exp_label": "MetaRLBO-Ising50-MLP-016",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 5.0,
    "outer_lr": 0.5,
    "num_meta_updates_per_iter": 30,
    "entropy_reg_coeff": 0.5,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising50_017 = {  # Copy general configs from AMP-KNR-003
    "exp_label": "MetaRLBO-Ising50-MLP-017",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 5000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 30,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising50_018 = {  # Increase number of meta updates
    "exp_label": "MetaRLBO-Ising50-MLP-018",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 5000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 40,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising50_019 = {  # Decrease beta
    "exp_label": "MetaRLBO-Ising50-MLP-019",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 5000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 40,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
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
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



metarlbo_ising50_020 = {  # Decrease diversity threshold
    "exp_label": "MetaRLBO-Ising50-MLP-020",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 5000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 40,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
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
        "diversity_threshold": 4, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising50_021 = {  # Decrease diversity threshold further
    "exp_label": "MetaRLBO-Ising50-MLP-021",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 5000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 40,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
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

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}



metarlbo_ising50_022 = {  # Copy KNR-008 -> (Make into MLP version) -- Direct comparison between MLP and KNR
    "exp_label": "MetaRLBO-Ising50-MLP-022",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 30,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.0, # Diversity hyperparameter -- higher is more penalty for more similar mols. -- essentially no penalty...
        "radius": 2, 
    },


    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising50_023 = {  # Copy 22 -> Increase learning rates
    "exp_label": "MetaRLBO-Ising50-MLP-023",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 5.0,
    "outer_lr": 0.5,
    "num_meta_updates_per_iter": 30,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.0, # Diversity hyperparameter -- higher is more penalty for more similar mols. -- essentially no penalty...
        "radius": 2, 
    },


    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising50_024 = {  # Copy 22 -> (increase number of meta updates)
    "exp_label": "MetaRLBO-Ising50-MLP-024",
    "task": "AltIsing50-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 2.0,
    "outer_lr": 0.2,
    "num_meta_updates_per_iter": 70,
    "entropy_reg_coeff": 0.0,
    "proxy_oracle": {
        "model_name": "MLP",
        "p": 0.8, 
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB", 
        "config": {
            'beta': 4.0,
        },
        "diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.0, # Diversity hyperparameter -- higher is more penalty for more similar mols. -- essentially no penalty...
        "radius": 2, 
    },


    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}
