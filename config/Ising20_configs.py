
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