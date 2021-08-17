

amp_000 = {
	"exp_label": "AMP-XGBoost",
	"num_proxies": 4, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 100,
	"num_samples_per_iter": 10, 
	"num_samples_per_task_update": 16, 
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


amp_random = {
	"exp_label": "AMP-RANDOM",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"policy": {
		"model_name": "RANDOM",
	},


	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 73,
}


# ======================= Proxy Oracles: K Nearest Regressors

amp_knr_001 = {
	"exp_label": "AMP-KNR-001",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4,
	"num_samples_per_task_update": 16, 
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_002 = { # incr. inner updates
	"exp_label": "AMP-KNR-002",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8,
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


amp_knr_003 = {
	"exp_label": "AMP-KNR-003",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4,
	"num_samples_per_task_update": 16, 
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
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

amp_knr_004 = {
	"exp_label": "AMP-KNR-004",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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


amp_knr_005 = {
	"exp_label": "AMP-KNR-005",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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


amp_knr_006 = {
	"exp_label": "AMP-KNR-006",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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

amp_knr_007 = {
	"exp_label": "AMP-KNR-007",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 1e-2,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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

amp_knr_008 = {
	"exp_label": "AMP-KNR-008",
	"num_proxies": 6, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 8, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 1e-2,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.9, # Proportion of data to sample to train proxy oracles
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

amp_knr_009 = {
	"exp_label": "AMP-KNR-009",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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


amp_knr_010 = {
	"exp_label": "AMP-KNR-010",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_011 = {
	"exp_label": "AMP-KNR-011",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.05, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_012 = {
	"exp_label": "AMP-KNR-012",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "XGBoost",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.05, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_013 = {
	"exp_label": "AMP-KNR-013",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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


amp_knr_014 = {
	"exp_label": "AMP-KNR-014",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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

amp_knr_015 = {
	"exp_label": "AMP-KNR-015",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 5.0,
	"outer_lr": 1.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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

amp_knr_016 = {
	"exp_label": "AMP-KNR-016",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 5.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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



amp_knr_010_2 = {
	"exp_label": "AMP-KNR-010-2",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 2,
}


amp_knr_011_2 = {
	"exp_label": "AMP-KNR-011-2",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.05, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 2,
}


amp_knr_012_2 = {
	"exp_label": "AMP-KNR-012-2",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "XGBoost",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.05, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 2,
}

amp_knr_013_2 = {
	"exp_label": "AMP-KNR-013-2",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 2,
}


amp_knr_014_2 = {
	"exp_label": "AMP-KNR-014-2",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 2,
}

amp_knr_015_2 = {
	"exp_label": "AMP-KNR-015-2",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 5.0,
	"outer_lr": 1.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 2,
}

amp_knr_016_2 = {
	"exp_label": "AMP-KNR-016-2",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 5.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 2,
}

amp_knr_010_3 = {
	"exp_label": "AMP-KNR-010-3",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 3,
}


amp_knr_011_3 = {
	"exp_label": "AMP-KNR-011-3",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.05, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 3,
}


amp_knr_012_3 = {
	"exp_label": "AMP-KNR-012-3",
	"num_proxies": 4, 
	"num_inner_updates": 2, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "XGBoost",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.05, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 3,
}

amp_knr_013_3 = {
	"exp_label": "AMP-KNR-013-3",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1e-1,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 3,
}


amp_knr_014_3 = {
	"exp_label": "AMP-KNR-014-3",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 3,
}

amp_knr_015_3 = {
	"exp_label": "AMP-KNR-015-3",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 5.0,
	"outer_lr": 1.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 3,
}

amp_knr_016_3 = {
	"exp_label": "AMP-KNR-016-3",
	"num_proxies": 4, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 5.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 3,
}




# Large Batches -- somewhat matching the DynaPPO paper

amp_knr_large_001 = {
	"exp_label": "AMP-KNR-Large-001",
	"num_proxies": 16, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 40,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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


amp_knr_large_002 = {
	"exp_label": "AMP-KNR-Large-002",
	"num_proxies": 16, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 50,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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


amp_knr_large_003 = {
	"exp_label": "AMP-KNR-Large-003",
	"num_proxies": 16, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16, 
	"num_samples_per_task_update": 16, 
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 40,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
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