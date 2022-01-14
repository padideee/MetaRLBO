
################ Baselines

dynappo_ising20_001 = {
    "exp_label": "dynappo_ising20_001",
	"task": "AltIsing20-v0",
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



# Small Batches -- Size 10

small_metarlbo_ising20_001 = { 
    "exp_label": "Small-MetaRLBO-Ising20-KNR-001",
    "task": "AltIsing20-v0",
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

small_metarlbo_ising20_002 = { 
    "exp_label": "Small-MetaRLBO-Ising20-KNR-002",
    "task": "AltIsing20-v0",
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

small_metarlbo_ising20_003 = { 
    "exp_label": "Small-MetaRLBO-Ising20-KNR-003",
    "task": "AltIsing20-v0",
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



metarlbo_ising20_001 = { 
    "exp_label": "MetaRLBO-Ising20-KNR-001",
    "task": "AltIsing20-v0",
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

metarlbo_ising20_002 = { 
    "exp_label": "MetaRLBO-Ising20-KNR-002",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
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

metarlbo_ising20_003 = { 
    "exp_label": "MetaRLBO-Ising20-KNR-003",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 3,
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


metarlbo_ising20_004 = { 
    "exp_label": "MetaRLBO-Ising20-KNR-004",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
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
        "diversity_threshold": 5, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising20_005 = { 
    "exp_label": "MetaRLBO-Ising20-KNR-005",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
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
        "diversity_threshold": 5, # Diversity threshold when greedily selecting molecules...
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising20_006 = { # More query proxies than 001 (also we have entropy reg coeff -- non-zero)
    "exp_label": "MetaRLBO-Ising20-KNR-006",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.1,
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

metarlbo_ising20_007 = { # more number of proxies compared to 006
    "exp_label": "MetaRLBO-Ising20-KNR-007",
    "task": "AltIsing20-v0",
    "num_proxies": 8, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.1,
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

metarlbo_ising20_008 = { # Increased entropy reg. vs 007
    "exp_label": "MetaRLBO-Ising20-KNR-006",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.5,
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


metarlbo_ising20_009 = { # Increased entropy reg. vs 008
    "exp_label": "MetaRLBO-Ising20-KNR-006",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
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


metarlbo_ising20_010 = {  # Same as 004 -- but we adjust the radius and lambda...
    "exp_label": "MetaRLBO-Ising20-KNR-010",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 75,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
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
        "diversity_threshold": 5, # Diversity threshold when greedily selecting molecules...
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


metarlbo_ising20_011 = {  # Same as 004 -- but we adjust the radius and lambda...
    "exp_label": "MetaRLBO-Ising20-KNR-011",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_initial_samples": 500,
    "num_query_proxies": 32,
    "num_samples_per_proxy": 150,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
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
        "diversity_threshold": 5, # Diversity threshold when greedily selecting molecules...
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


metarlbo_ising20_012 = { # Same as 007 -- not sure if true (but with MLP instead...)
    "exp_label": "MetaRLBO-Ising20-MLP-012",
    "task": "AltIsing20-v0",
    "num_proxies": 8, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.1,
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



metarlbo_ising20_013 = { # Same as 012 (but with more inner loop updates)
    "exp_label": "MetaRLBO-Ising20-MLP-013",
    "task": "AltIsing20-v0",
    "num_proxies": 8, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.1,
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


metarlbo_ising20_014 = { # Copy 006 but use MLP 
    "exp_label": "MetaRLBO-Ising20-KNR-014",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.1,
    "proxy_oracle": {
        "model_name": "MLP",
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


metarlbo_ising20_015 = { # Same as 014 but use p=0.8 instead
    "exp_label": "MetaRLBO-Ising20-KNR-015",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.1,
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

metarlbo_ising20_016 = { # Same as 015 but with lower beta
    "exp_label": "MetaRLBO-Ising20-KNR-016",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.1,
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


metarlbo_ising20_017 = { # Same as 016 but lower num meta updates
    "exp_label": "MetaRLBO-Ising20-MLP-017",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 40, 
    "entropy_reg_coeff": 0.1,
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




metarlbo_ising20_018 = { # Same as 016 -- but with increased number of meta updates (50 -> 80) + increased number of max queries
    "exp_label": "MetaRLBO-Ising20-MLP-018",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 10000, # Maximum number of queries in experiment 
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.1,
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


metarlbo_ising20_019 = { # Same as 018 -- but change the diversity penalty a bit (radius 2 -> 1)... 
    "exp_label": "MetaRLBO-Ising20-MLP-019",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 10000, # Maximum number of queries in experiment 
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.1,
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 1, 
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_020 = { # Same as 019 -- but change the diversity penalty a bit more (lambda 0.1 -> 1)... 
    "exp_label": "MetaRLBO-Ising20-MLP-020",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 10000, # Maximum number of queries in experiment 
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.1,
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 1, 
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_021 = { # Same as 020 -- but increase the learning rate (5.0 and 0.5)
    "exp_label": "MetaRLBO-Ising20-MLP-021",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 10000, # Maximum number of queries in experiment 
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 5.0,
    "outer_lr": 0.5,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.1,
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 1, 
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_022 = { # Copy 006 but use CNN instead
    "exp_label": "MetaRLBO-Ising20-CNN-022",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.1,
    "proxy_oracle": {
        "model_name": "CNN",
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


metarlbo_ising20_023 = { # Same as 019 but lambda = 0 and proxy oracle is CNN
    "exp_label": "MetaRLBO-Ising20-CNN-023",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment 
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.1,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 1, 
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising20_024 = { # Same as 023 but no entropy reg coeff
    "exp_label": "MetaRLBO-Ising20-CNN-024",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment 
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.1,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 0.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 1, 
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising20_025 = { # Copy 022 but use more entropy bonus
    "exp_label": "MetaRLBO-Ising20-CNN-025",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 1.0,
    "proxy_oracle": {
        "model_name": "CNN",
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


metarlbo_ising20_026 = { # Copy 022 but use more entropy bonus
    "exp_label": "MetaRLBO-Ising20-CNN-026",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 5.0,
    "proxy_oracle": {
        "model_name": "CNN",
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


metarlbo_ising20_027 = { # Copy 022 but use more entropy bonus
    "exp_label": "MetaRLBO-Ising20-CNN-027",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 10.0,
    "proxy_oracle": {
        "model_name": "CNN",
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


metarlbo_ising20_028 = { # Copy 026 but use more lambda!
    "exp_label": "MetaRLBO-Ising20-CNN-028",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 5.0,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_029 = { # Copy 028 but with a larger lambda (1 -> 3)
    "exp_label": "MetaRLBO-Ising20-CNN-029",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 5.0,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 3.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_030 = { # Copy but with a larger lambda (1 -> 5)
    "exp_label": "MetaRLBO-Ising20-CNN-030",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 5.0,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 5.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_031 = { # Copy 029 but with a larger radius
    "exp_label": "MetaRLBO-Ising20-CNN-031",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 32,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 5.0,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 3.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 4, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_032 = { # New configs w/ fixed entropy reg coeff... also it's the mean of the envs now
    "exp_label": "MetaRLBO-Ising20-CNN-032",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 3.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 4, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_033 = {  # 032 -- but with larger outer lr
    "exp_label": "MetaRLBO-Ising20-CNN-033",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.5,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 3.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 4, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}

metarlbo_ising20_034 = {  # 032 but with more inner loop updates (1 -> 2)
    "exp_label": "MetaRLBO-Ising20-CNN-034",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 3.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 4, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_035 = {  # 032 (but more meta updates)
    "exp_label": "MetaRLBO-Ising20-CNN-035",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 3.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 4, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_036 = {  # 032 -- but with more entropy 0.2 -> 1
    "exp_label": "MetaRLBO-Ising20-CNN-036",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.5,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 1.0,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 3.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 4, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_037 = {  # 032 -- but with more entropy 0.2 -> 5
    "exp_label": "MetaRLBO-Ising20-CNN-037",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.5,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 5.0,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 3.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 4, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_038 = { # 032 -- but with lower radius (4 -> 1)
    "exp_label": "MetaRLBO-Ising20-CNN-038",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_039 = { # Testing with baseline 
    "exp_label": "MetaRLBO-Ising20-CNN-039",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
        "radius": 2, 
    },

    "true_oracle": {
        "model_name": "AltIsing_Oracle",
    },
    "use_baseline": True,
    "log_interval": 1,
    "results_log_dir": "./logs",
    "seed": 73,
}


metarlbo_ising20_040 = { # Testing without baseline 
    "exp_label": "MetaRLBO-Ising20-CNN-040",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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



metarlbo_ising20_041 = { # Copy 039 - Entropy reg. coeff (0.2 -> 0.5)
    "exp_label": "MetaRLBO-Ising20-CNN-041",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.5,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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


metarlbo_ising20_042 = { # Copy 039 - Entropy Reg. Coeff (0.2 -> 1.0)
    "exp_label": "MetaRLBO-Ising20-CNN-042",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 1.0,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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



metarlbo_ising20_043 = { # Copy 041 - More inner updates (1 -> 2)
    "exp_label": "MetaRLBO-Ising20-CNN-043",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 2,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.5,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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



metarlbo_ising20_044 = { # 41 with more updates w/ more updates
    "exp_label": "MetaRLBO-Ising20-CNN-044",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.5,
    "proxy_oracle": {
        "model_name": "CNN",
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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

metarlbo_ising20_045 = { # Copy 041 w/ p (0.7 -> 0.5)
    "exp_label": "MetaRLBO-Ising20-CNN-045",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.5,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.5, 
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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


metarlbo_ising20_046 = { # 41 with lower beta
    "exp_label": "MetaRLBO-Ising20-CNN-046",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.5,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, 
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
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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


metarlbo_ising20_047 = { # Copy 040 (w/ p: 0.7 -> 0.5)
    "exp_label": "MetaRLBO-Ising20-CNN-047",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.5, 
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
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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


metarlbo_ising20_048 = { # Copy 040 (w/ beta: 4 -> 2)
    "exp_label": "MetaRLBO-Ising20-CNN-048",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, 
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
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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

metarlbo_ising20_049 = { # Copy 040 (w/ num_query_proxies: 32 -> 64)
    "exp_label": "MetaRLBO-Ising20-CNN-049",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 64,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, 
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
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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

metarlbo_ising20_050 = { # Copy 040 (w/ num_samples_per_proxy: 64 -> 128)
    "exp_label": "MetaRLBO-Ising20-CNN-050",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 128,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, 
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
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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

metarlbo_ising20_051 = { # Copy 040 (w/ lambda 1.0 -> 2.0)
    "exp_label": "MetaRLBO-Ising20-CNN-051",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, 
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
        "lambda": 2.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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


metarlbo_ising20_052 = { # Copy 040 (w/ num_meta_updates_per_iter 50 -> 80)
    "exp_label": "MetaRLBO-Ising20-CNN-052",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, 
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
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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


metarlbo_ising20_053 = { # Copy 050 (w/ num_samples_per_proxy: 128 -> 256)
    "exp_label": "MetaRLBO-Ising20-CNN-053",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, 
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
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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



metarlbo_ising20_054 = { # Copy 053 w/ (num_meta_updates_per_iter 50 -> 80)
    "exp_label": "MetaRLBO-Ising20-CNN-054",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 256,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, 
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
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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



metarlbo_ising20_055 = { # Copy 050 (w/ num_samples_per_proxy: 256 -> 384)
    "exp_label": "MetaRLBO-Ising20-CNN-055",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 384,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 50, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, 
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
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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



metarlbo_ising20_056 = { # Copy 052 (w/ UCB beta: 2.0 -> 0.0) -- i.e. Posterior Mean
    "exp_label": "MetaRLBO-Ising20-CNN-056",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 80, 
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 0.7, 
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
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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




metarlbo_ising20_057 = { # Copy 052 (w/ p 0.7 -> 1.0)
    "exp_label": "MetaRLBO-Ising20-CNN-057",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
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
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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

metarlbo_ising20_058 = { # Copy 057 (w/ beta 2 -> 1)
    "exp_label": "MetaRLBO-Ising20-CNN-058",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
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
            'beta': 1.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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





metarlbo_ising20_059 = { # Copy 057 (w/ beta 2 -> 0)
    "exp_label": "MetaRLBO-Ising20-CNN-059",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
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
            'beta': 0.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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




metarlbo_ising20_060 = { # Copy 057 (w/ lambda 1.0 -> 2.0)
    "exp_label": "MetaRLBO-Ising20-CNN-060",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
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
        "lambda": 2.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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

metarlbo_ising20_061 = { # Copy 057 (w/ lambda 1.0 -> 3.0)
    "exp_label": "MetaRLBO-Ising20-CNN-061",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
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
        "lambda": 3.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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


metarlbo_ising20_062 = { # Copy 059 -- incr entr reg coeff (0.2 -> 0.5)
    "exp_label": "MetaRLBO-Ising20-CNN-062",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
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
            'beta': 0.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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


metarlbo_ising20_063 = { # Copy 059 -- beta 0 -> 1
    "exp_label": "MetaRLBO-Ising20-CNN-063",
    "task": "AltIsing20-v0",
    "num_proxies": 4, 
    "max_num_queries": 8000, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 32,
    "num_initial_samples": 500,
    "num_samples_per_proxy": 64,
    "num_query_per_iter": 500,
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
            'beta': 1.0,
        },
        "diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
    },
    "env": { # See DynaPPO paper for these configs
        "lambda": 1.0, # Diversity hyperparameter -- higher is more penalty for more similar mols.
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
