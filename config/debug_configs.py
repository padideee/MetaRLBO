



debug = {
	"exp_label": "DEBUG",
	"num_proxies": 2, 
	"num_inner_updates": 3, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 10,
	"num_samples_per_proxy": 4, 
	"num_samples_per_task_update": 4, 
	"proxy_oracle": {
		"model_name": "BR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
}

debug_RANDOM = {
	"exp_label": "DEBUG-RANDOM",
	"num_initial_samples": 400,
	"num_query_per_iter": 250,
	"policy": {
		"model_name": "RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"log_interval": 1,
}

debug_TRPO = {
	"exp_label": "DEBUG-TRPO",
	"num_proxies": 2,
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 40, 
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 40,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 20,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 2.0,
		}
	},
	"metalearner": {
		"method": "TRPO",  # REINFORCE or TRPO
		"tau": 1.0,
		"gamma": 0.99,
		"max_kl": 1e-2,
		"cg_iters": 10,
		"cg_damping": 1e-5,
		"ls_max_steps": 15,
		"ls_backtrack_ratio": 0.8,
	},
	"num_processes": 4,
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}

debug_TRPO_True = {
	"exp_label": "DEBUG-TRPO-True",
	"num_proxies": 2,
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 8, 
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 20.0,
	"outer_lr": 2.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 2.0,
		}
	},
	"outerloop_oracle": "true",  # Options: proxy, true
	"metalearner": {
		"method": "TRPO",  # REINFORCE or TRPO
		"tau": 1.0,
		"gamma": 0.99,
		"max_kl": 1e-2,
		"cg_iters": 10,
		"cg_damping": 1e-5,
		"ls_max_steps": 15,
		"ls_backtrack_ratio": 0.8,
	},
	"num_processes": 4,
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}




debug_REINFORCE = {
	"exp_label": "DEBUG-REINFORCE",
	"num_proxies": 4,
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 16, 
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 10,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 6.0,
		}
	},
	"metalearner": {
		"method": "REINFORCE",  # REINFORCE or TRPO
	},
	"num_processes": 8,
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}

debug_REINFORCE_True = {
	"exp_label": "DEBUG-REINFORCE-True",
	"num_proxies": 2,
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 8, 
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 10.0,
	"outer_lr": 1.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 2.0,
		}
	},
	"outerloop_oracle": "true",  # Options: proxy, true
	"metalearner": {
		"method": "REINFORCE",  # REINFORCE or TRPO
	},
	"num_processes": 4,
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}

debug_REINFORCE_True_BR = {
	"exp_label": "DEBUG-REINFORCE-True-BR",
	"num_proxies": 2,
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 8, 
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 10.0,
	"outer_lr": 1.0,
	"proxy_oracle": {
		"model_name": "BR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 2.0,
		}
	},
	"outerloop_oracle": "true",  # Options: proxy, true
	"metalearner": {
		"method": "REINFORCE",  # REINFORCE or TRPO
	},
	"num_processes": 4,
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}






# Varying Proxy Oracle Models
debug_BR = {
	"exp_label": "DEBUG-BR",
	"num_proxies": 2, 
	"num_inner_updates": 3, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 10,
	"num_samples_per_proxy": 4, 
	"num_samples_per_task_update": 4, 
	"proxy_oracle": {
		"model_name": "BR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
}



debug_RR = {
	"exp_label": "DEBUG-RR",
	"num_proxies": 2, 
	"num_inner_updates": 3, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 10,
	"num_samples_per_proxy": 4, 
	"num_samples_per_task_update": 4, 
	"proxy_oracle": {
		"model_name": "BR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
}




debug_GPR_RBF = {
	"exp_label": "DEBUG-GPR-RBF",
	"num_proxies": 2,
	"num_inner_updates": 1, 
	"num_initial_samples": 250,
	"num_query_proxies": 4,
	"num_samples_per_proxy": 30, 
	"num_query_per_iter": 20,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
		"config": {
			"kernel": "RBF",
		},
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"metalearner": {
		"method": "REINFORCE",  # REINFORCE or TRPO
	},
	"num_processes": 4,
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}

debug_GPR_RQ = {
	"exp_label": "DEBUG-GPR-RQ",
	"num_proxies": 4,
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 8, 
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
		"config": {
			"kernel": "RationalQuadratic",
		},
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 6.0,
		}
	},
	"metalearner": {
		"method": "REINFORCE",  # REINFORCE or TRPO
	},
	"num_processes": 4,
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}
debug_GPR_M = {
	"exp_label": "DEBUG-GPR-M",
	"num_proxies": 2,
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 8, 
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 20.0,
	"outer_lr": 2.0,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
		"config": {
			"kernel": "Matern",
		},
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 6.0,
		}
	},
	"metalearner": {
		"method": "REINFORCE",  # REINFORCE or TRPO
	},
	"num_processes": 4,
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}



debug_RFR = {
	"exp_label": "DEBUG-RFR",
	"num_proxies": 4,
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 100,
	"num_query_proxies": 8,
	"num_samples_per_proxy": 16, 
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 10,
	"inner_lr": 7.0,
	"outer_lr": 0.7,
	"proxy_oracle": {
		"model_name": "RFR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 6.0,
		}
	},
	"num_processes": 8,
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}

debug_KNR = {
	"exp_label": "DEBUG-KNR",
	"num_proxies": 2, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.2,
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
		"diversity_threshold": 0, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"reset_policy_per_round": True,
	"log_interval": 1,
	"results_log_dir": "./logs",
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"seed": 73,
}

debug_RNA14 = {
	"exp_label": "DEBUG-RNA14",
    "task": "RNA14-v0",
	"num_proxies": 2, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16,
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.2,
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
		"diversity_threshold": 0, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RNA14_Oracle",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"seed": 73,
}

debug_med_Ising50 = {
	"task": "AltIsing50-v0",
	"exp_label": "DEBUG-med-Ising50",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4, 
	"num_query_proxies": 4,
	"num_initial_samples": 500,
	"num_samples_per_proxy": 16,
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
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
		"diversity_threshold": 0, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "AltIsing_Oracle",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"seed": 73,
}

debug_metarlbo_Ising20 = {
	"task": "AltIsing20-v0",
	"exp_label": "DEBUG-med-Ising20",
	"num_proxies": 2, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, 
	"num_query_proxies": 4,
	"num_initial_samples": 500,
	"num_samples_per_proxy": 16,
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 2, 
	"entropy_reg_coeff": 0.5, # Leo: TODO - test this!
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
		"diversity_threshold": 0, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "AltIsing_Oracle",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"seed": 73,
}


debug_metarlbo_Ising50 = {
	"task": "AltIsing50-v0",
	"exp_label": "DEBUG-med-Ising50",
	"num_proxies": 2, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, 
	"num_query_proxies": 4,
	"num_initial_samples": 500,
	"num_samples_per_proxy": 16,
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 2, 
	"entropy_reg_coeff": 0.5, # Leo: TODO - test this!
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
		"diversity_threshold": 0, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "AltIsing_Oracle",
	},
	"use_baseline": False, # Use a linear baseline or no...
	"log_interval": 1,
	"results_log_dir": "./logs",
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"seed": 73,
}


debug_metarlbo_RNA14 = {
	"task": "RNA14-v0",
	"exp_label": "DEBUG-RNA14",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, 
	"num_query_proxies": 4,
	"num_initial_samples": 500,
	"num_samples_per_proxy": 16,
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
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
			'beta': 4.0,
		},
		"diversity_threshold": 0, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "AltIsing_Oracle",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"seed": 73,
}



debug_mid = {
	"exp_label": "DEBUG-Medium",
	"num_proxies": 8, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_samples_per_proxy": 16, 
	"num_samples_per_task_update": 16, 
	"num_query_per_iter": 100,
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 4,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 1, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

# Varying Policy Models

debug_GRU = {
	"exp_label": "DEBUG",
	"num_proxies": 2, 
	"num_inner_updates": 3, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 10,
	"num_samples_per_proxy": 4, 
	"num_samples_per_task_update": 4, 
	"proxy_oracle": {
		"model_name": "BR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"policy": {
		"model_name": "GRU",
		"model_config": {
			"hidden_dim": 100,
			"state_embedding_size": 64,
		}
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
}

debug_diversity = {
	"exp_label": "DEBUG",
	"num_proxies": 2,
	"num_inner_updates": 3,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 10,
	"num_samples_per_proxy": 4,
	"num_samples_per_task_update": 4,
	"proxy_oracle": {
		"model_name": "BR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},

	"diversity": {
		"div_metric_name": "blast",
		"div_switch": "ON"
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
}









######################### CLAMP DEBUGGING


debug_CLAMP = {
	"exp_label": "DEBUG-CLAMP",
	"task": "CLAMP-v0",
	"max_num_queries": 20,
	"num_proxies": 2,
	"num_inner_updates": 1, 
	"num_initial_samples": 10,
	"num_query_proxies": 2,
	"num_samples_per_proxy": 8, 
	"num_query_per_iter": 5,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 6.0,
		}
	},
	"metalearner": {
		"method": "REINFORCE",  # REINFORCE or TRPO
	},
	"CLAMP": { # CLAMP Specific configs... do not use normally
		"true_oracle_model": "RandomForest", # RandomForest or MLP
		"data_source": "D1_target",
		"use_pretrained_model": False,
		"evaluation": { # post-training
			"num_query_proxies": 2,
			"num_samples_per_proxy": 4,
			"num_mols_select": 5,
		},
	},
	"num_processes": 1,
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}


debug_CLAMP_GFN = {
	"exp_label": "DEBUG-CLAMP-GFN",
	"task": "CLAMP-v0",
	"max_num_queries": 20,
	"num_proxies": 2,
	"num_inner_updates": 1, 
	"num_initial_samples": 10,
	"num_query_proxies": 2,
	"num_samples_per_proxy": 8, 
	"num_query_per_iter": 5,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 6.0,
		}
	},
	"metalearner": {
		"method": "REINFORCE",  # REINFORCE or TRPO
	},
	"CLAMP": { # CLAMP Specific configs... do not use normally
		"true_oracle_model": "GFN", # RandomForest or MLP or GFN
		"evaluation": { # post-training
			"num_query_proxies": 2,
			"num_samples_per_proxy": 4,
			"num_mols_select": 5,
		},
	},
	"num_processes": 1,
	"mode": "test", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}


debug_dynappo = {
    "exp_label": "dynappo_debug",
    "task": "AMP-v0",
    "use_metalearner": False,
    "max_num_queries": 3000, # Maximum number of queries in experiment
    "query_storage_size": 100000, # Maximum number of queries allowed in the storage
    "num_updates_per_iter": 72,
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
        "num_steps": 150,
    },  
    "save_interval": 10, # Save model every n batch queries
    "num_processes": 8, 
    "results_log_dir": "./logs", 
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
    "seed": 73,
}


debug_dynappo_ising20 = {
    "exp_label": "dynappo_debug_ising20",
	"task": "AltIsing20-v0",
    "use_metalearner": False,
    "max_num_queries": 3000, # Maximum number of queries in experiment
    "query_storage_size": 100000, # Maximum number of queries allowed in the storage
    "num_updates_per_iter": 72,
	"num_initial_samples": 500,
	"num_query_per_iter": 500,
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
        "num_steps": 150,
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




debug_metarlbo_ising20_trpo = {
    "exp_label": "debug_metarlbo_ising20_trpo",
	"task": "AltIsing20-v0",
	"num_proxies": 2, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, 
	"num_query_proxies": 4,
	"num_initial_samples": 500,
	"num_samples_per_proxy": 16,
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 20,
	"inner_lr": 0.1,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"metalearner": {
		"method": "TRPO",  # REINFORCE or TRPO
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},
	"metalearner": {
		"method": "TRPO",  # REINFORCE or TRPO
		"tau": 1.0,
		"gamma": 1.0,
		"max_kl": 1e-2,
		"cg_iters": 10,
		"cg_damping": 1e-5,
		"ls_max_steps": 15,
		"ls_backtrack_ratio": 0.8,
	},
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB", 
		"config": {
			'beta': 4.0,
		},
		"diversity_threshold": 0, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "AltIsing_Oracle",
	},
	"num_processes": 8, 
	"log_interval": 1,
	"results_log_dir": "./logs",
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"seed": 73,
}


debug_Thesis_rna14_026_test_2 = { # RND on
    "exp_label": "DEBUG_Thesis_MetaRLBO-RNA14-026-test_2",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 4,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 16,
    "num_query_per_iter": 20,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 1,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN", #CNN #CNN_dropout
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

debug_Thesis_rna14_026_test_3 = { # RND on, only IN reward
    "exp_label": "DEBUG_Thesis_MetaRLBO-RNA14-026-test_3",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 4,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 16,
    "num_query_per_iter": 20,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 1,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN", #CNN #CNN_dropout
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

debug_Thesis_rna14_026_test_4 = { # RND on, only EX reward
    "exp_label": "DEBUG_Thesis_MetaRLBO-RNA14-026-test_4",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 4,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 16,
    "num_query_per_iter": 20,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 1,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN", #CNN #CNN_dropout
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
	"reward" : "E",
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

debug_Thesis_rna14_026_test_2_lambda01 = { # RND on, lambda: 0.1
    "exp_label": "DEBUG_Thesis_MetaRLBO-RNA14-026-test_2_lambda01",
    "task": "RNA14-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment
    "num_inner_updates": 1,
    "num_query_proxies": 4,
    "num_initial_samples": 100,
    "num_samples_per_proxy": 16,
    "num_query_per_iter": 20,
    "inner_lr": 1.0,
    "outer_lr": 0.1,
    "num_meta_updates_per_iter": 1,
    "entropy_reg_coeff": 0.2,
    "proxy_oracle": {
        "model_name": "CNN", #CNN #CNN_dropout
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

# -------------------------
debug_metarlbo_ising20_052 = { # Copy 040 (w/ num_meta_updates_per_iter 50 -> 80)
    "exp_label": "DEBUG_MetaRLBO-Ising20-CNN-052",
    "task": "AltIsing20-v0",
    "num_proxies": 4,
    "max_num_queries": 1500, # Maximum number of queries in experiment #?
    "num_inner_updates": 1,
    "num_query_proxies": 4,
    "num_initial_samples": 100, #250
    "num_samples_per_proxy": 16,
    "num_query_per_iter": 20, #250
    "inner_lr": 1.0, #2.0
    "outer_lr": 0.1, #0.2
    "num_meta_updates_per_iter": 1,
    "entropy_reg_coeff": 0.2, #0.0
    "proxy_oracle": {
        "model_name": "CNN",
        "p": 1.0, #0.7
    },
    "outerloop": {
        "oracle": "proxy",
        "density_penalty": True,
    },
    "selection_criteria": { # Configs for selecting the samples
        "method": "UCB",
        "config": {
            'beta': 2.0, #4.0
        },
        "diversity_threshold": 1, #10 # Diversity threshold when greedily selecting molecules...
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
	"reward" : "E+IN",
	"diversity": {
            "div_metric_name": "RND", # Options: "hamming" or "blast" or "RND" (Note: blast is slow)
            "div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
    },
	"exp_policy": {
            "model_name": "MLP",
            "input_size": 400,  # 20*20
            "hidden_size": 200,  # 20 * 10
            "output_size": 20,
    },
}