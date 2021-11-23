## Baseline configs...


# Dynappo configs

dynappo_amp_001 = {
    "exp_label": "dynappo_amp_001",
    "task": "AMP-v0",
    "use_metalearner": False,
    "max_num_queries": 3000, # Maximum number of queries in experiment
    "query_storage_size": 100000, # Maximum number of queries allowed in the storage
    "num_updates_per_iter": 72,
    "num_initial_samples": 250,
    "num_query_per_iter": 250,
    'num_samples_per_iter': 300,
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
    "save_interval": 10, # Save model every n batch queries
    "num_processes": 8, 
    "results_log_dir": "./logs", 
    "seed": 73,
}


#### Meta-learning configs

amp_000 = {
	"exp_label": "AMP-XGBoost",
	"num_proxies": 4,
	"num_initial_samples": 100,
	"num_samples_per_proxy": 10,
	"proxy_oracle": {
		"model_name": "XGBoost",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"policy": {
		"hidden_dim": 100,
		"state_embedding_size": 64,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_random_large = {
	"exp_label": "AMP-RANDOM-Large",
	"num_initial_samples": 250,
	"num_query_per_iter": 250,
	"policy": {
		"model_name": "RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 40,
	"log_interval": 1,
}

amp_random_med = {
	"exp_label": "AMP-RANDOM-Medium",
	"num_initial_samples": 250,
	"num_query_per_iter": 50,
	"policy": {
		"model_name": "RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 70,
	"log_interval": 1,
}
amp_random = {
	"exp_label": "AMP-RANDOM-Medium",
	"num_initial_samples": 250,
	"num_query_per_iter": 10,
	"policy": {
		"model_name": "RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 73,
	"log_interval": 1,
}

amp_dynappo_random_large = {
	"exp_label": "AMP-DynaPPO_RANDOM-Large",
	"num_initial_samples": 250,
	"num_query_per_iter": 250,
	"policy": {
		"model_name": "DynaPPO_RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 50,
	"log_interval": 1,
}

amp_dynappo_random_med = {
	"exp_label": "AMP-DynaPPO_RANDOM-Medium",
	"num_initial_samples": 250,
	"num_query_per_iter": 50,
	"policy": {
		"model_name": "DynaPPO_RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 70,
	"log_interval": 1,
}
amp_dynappo_random = {
	"exp_label": "AMP-DynaPPO_RANDOM-Medium",
	"num_initial_samples": 250,
	"num_query_per_iter": 10,
	"policy": {
		"model_name": "DynaPPO_RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 73,
	"log_interval": 1,
}




# ======================= Proxy Oracles: K Nearest Regressors



metarlbo_amp_knr_001 = {
	"exp_label": "MetaRLBO-AMP-KNR-001",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

metarlbo_amp_knr_002 = {
	"exp_label": "MetaRLBO-AMP-KNR-002",
	"num_proxies": 4,
	"num_inner_updates": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


metarlbo_amp_knr_003 = {
	"exp_label": "MetaRLBO-AMP-KNR-003",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

metarlbo_amp_knr_004 = { 
	"exp_label": "MetaRLBO-AMP-KNR-004",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

metarlbo_amp_knr_005 = {
	"exp_label": "MetaRLBO-AMP-KNR-005",
	"num_proxies": 4,
	"num_inner_updates": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

metarlbo_amp_knr_006 = { 
	"exp_label": "MetaRLBO-AMP-KNR-006",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

metarlbo_amp_knr_007 = {
	"exp_label": "MetaRLBO-AMP-KNR-007",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

metarlbo_amp_knr_008 = { 
	"exp_label": "MetaRLBO-AMP-KNR-008",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 30,
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
		"diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}







################# Using MLP as the oracle


metarlbo_amp_mlp_001 = {
	"exp_label": "MetaRLBO-AMP-MLP-001",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

metarlbo_amp_mlp_002 = { # 2 inner loop updates (compared to 001)
	"exp_label": "MetaRLBO-AMP-MLP-002",
	"num_proxies": 4,
	"num_inner_updates": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


metarlbo_amp_mlp_003 = { # More proxies (compared to 001)
	"exp_label": "MetaRLBO-AMP-MLP-003",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}




# ======================= Old configs


amp_knr_001 = { # Increased number of proxy models...
	"exp_label": "AMP-KNR-001",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_002 = { # Increased number of proxy models...
	"exp_label": "AMP-KNR-002",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_003 = { # Increased number of proxy models...
	"exp_label": "AMP-KNR-003",
	"num_proxies": 4, 
	"num_inner_updates": 3,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_004 = { # Increased number of proxy models...
	"exp_label": "AMP-KNR-004",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_005 = { # Increased number of proxy models...
	"exp_label": "AMP-KNR-005",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 10.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_006 = { 
	"exp_label": "AMP-KNR-006",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "true",
		"density_penalty": False,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_007 = { 
	"exp_label": "AMP-KNR-007",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_008 = { 
	"exp_label": "AMP-KNR-008",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "true",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_009 = { 
	"exp_label": "AMP-KNR-009",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_010 = { 
	"exp_label": "AMP-KNR-010",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 10.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_011 = { 
	"exp_label": "AMP-KNR-011",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_012 = { 
	"exp_label": "AMP-KNR-012",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 2, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_013 = { 
	"exp_label": "AMP-KNR-013",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 8,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_014 = { 
	"exp_label": "AMP-KNR-014",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 16,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_015 = { 
	"exp_label": "AMP-KNR-015",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_016 = { 
	"exp_label": "AMP-KNR-016",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_017 = { 
	"exp_label": "AMP-KNR-017",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_018 = { 
	"exp_label": "AMP-KNR-018",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
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
			'beta': 8.0,
		},
		"diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_019 = { 
	"exp_label": "AMP-KNR-019",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 8,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_020 = { 
	"exp_label": "AMP-KNR-020",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 3, 
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_021 = { 
	"exp_label": "AMP-KNR-021",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_022 = { 
	"exp_label": "AMP-KNR-022",
	"num_proxies": 2, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_023 = { 
	"exp_label": "AMP-KNR-023",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 10,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
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
			'beta': 8.0,
		},
		"diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_024 = { 
	"exp_label": "AMP-KNR-024",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
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
			'beta': 0.0,
		},
		"diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_025 = { 
	"exp_label": "AMP-KNR-025",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 8,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 8,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
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
			'beta': 8.0,
		},
		"diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_026 = { 
	"exp_label": "AMP-KNR-026",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 1, 
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_027 = { 
	"exp_label": "AMP-KNR-027",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 3, 
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}



# ======================= Proxy Oracles: Bayesian Regression


amp_br_001 = { # Increased number of proxy models...
	"exp_label": "AMP-BR-001",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "BR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},


	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_br_002 = { # Increased number of proxy models...
	"exp_label": "AMP-BR-002",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},


	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

# ======================= Proxy Oracles: Gaussian Process Regressor -- All Large batches now!



# =========== Rational Quadratic
amp_gpr_rq_001 = { 
	"exp_label": "AMP-GPR_RQ-001",
	"max_num_queries": 2000,
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 30,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, 
		"config": {
			"kernel": "RationalQuadratic",
		},
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}
amp_gpr_rq_002 = { 
	"exp_label": "AMP-GPR_RQ-002",
	"max_num_queries": 2000,
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 30,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, 
		"config": {
			"kernel": "RationalQuadratic",
		},
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}



# =========== Matern
amp_gpr_m_001 = { 
	"exp_label": "AMP-GPR_M-001",
	"max_num_queries": 2000,
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 30,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, 
		"config": {
			"kernel": "Matern",
		},
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}
amp_gpr_m_002 = { 
	"exp_label": "AMP-GPR_M-002",
	"max_num_queries": 2000,
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 30,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, 
		"config": {
			"kernel": "Matern",
		},
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}






# =========== RBF
amp_gpr_rbf_001 = { 
	"exp_label": "AMP-GPR_RBF-001",
	"max_num_queries": 2000,
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 30,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, 
		"config": {
			"kernel": "RBF",
		},
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_gpr_rbf_002 = { 
	"exp_label": "AMP-GPR_RBF-002",
	"max_num_queries": 2000,
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 30,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, 
		"config": {
			"kernel": "RBF",
		},
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}






























#######################################################################################################################################

# Medium Batches -- smaller batches than DynaPPO paper but still large

amp_knr_med_001 = { 
	"exp_label": "AMP-KNR-Medium-001",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 16,
	"num_query_per_iter": 100,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 5,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_002 = { 
	"exp_label": "AMP-KNR-Medium-002",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 4,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 5,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_003 = { 
	"exp_label": "AMP-KNR-Medium-003",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 4,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 5,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_004 = { 
	"exp_label": "AMP-KNR-Medium-004",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 37,
	"num_query_per_iter": 100,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 5,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_005 = { 
	"exp_label": "AMP-KNR-Medium-005",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 4,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 10,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_006 = { 
	"exp_label": "AMP-KNR-Medium-006",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 4,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 5,
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
		"diversity_threshold": 30, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_007 = { 
	"exp_label": "AMP-KNR-Medium-007",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 37,
	"num_query_per_iter": 100,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 5,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_008 = { 
	"exp_label": "AMP-KNR-Medium-008",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 4,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 5,
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
			'beta': 8.0,
		},
		"diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_009 = { 
	"exp_label": "AMP-KNR-Medium-009",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 10,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_010 = { 
	"exp_label": "AMP-KNR-Medium-010",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 20,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_011 = { 
	"exp_label": "AMP-KNR-Medium-011",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_012 = { 
	"exp_label": "AMP-KNR-Medium-012",
	"num_proxies": 4,
	"num_inner_updates": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 10,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_013 = { 
	"exp_label": "AMP-KNR-Medium-013",
	"num_proxies": 4,
	"num_inner_updates": 3,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 10,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_014 = { 
	"exp_label": "AMP-KNR-Medium-014",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 10,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}








































##################################################################################################################################
# Large Batches -- somewhat matching the DynaPPO paper

amp_knr_large_001 = {
	"exp_label": "AMP-KNR-Large-001",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_002 = {
	"exp_label": "AMP-KNR-Large-002",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
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
		"diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_003 = {
	"exp_label": "AMP-KNR-Large-003",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 128,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 30,
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
		"diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_004 = { 
	"exp_label": "AMP-KNR-Large-004",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_005 = {
	"exp_label": "AMP-KNR-Large-005",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
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
			'beta': 8.0,
		},
		"diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_006 = {
	"exp_label": "AMP-KNR-Large-006",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 60,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_007 = {
	"exp_label": "AMP-KNR-Large-007",
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 0.5,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 120,
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
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

########## start: diversity configs
amp_knr_ham_med_003 = {
	"exp_label": "AMP-KNR-Medium-003",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 32,
	"num_query_per_iter": 100,
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 10,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
		"metric": "hamming"
	},
	"diversity": {
		"div_metric_name": "hamming",
		"div_switch": "ON"
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_blast_med_003 = {
	"exp_label": "AMP-KNR-Medium-003",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 32,
	"num_query_per_iter": 100,
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 10,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
		"metric": "hamming"
	},

	"diversity": {
		"div_metric_name": "blast",
		"div_switch": "ON"
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_off_med_003 = {
	"exp_label": "AMP-KNR-Medium-003",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 32,
	"num_query_per_iter": 100,
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 10,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
		"metric": "hamming"
	},

	"diversity": {
		"div_metric_name": "blast",
		"div_switch": "OFF"
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 73,
}

########## end: diversity configs

























