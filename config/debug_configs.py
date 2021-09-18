



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
	"num_proxies": 2,
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



debug_KNR = {
	"exp_label": "DEBUG-KNR",
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
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},
	"entropy_reg_coeff": 0.0,
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 6.0,
		}
	},
	"num_processes": 4,
	"mode": "val", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1,
}

debug_GPR_RBF = {
	"exp_label": "DEBUG-GPR-RBF",
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
			"kernel": "RBF",
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
		"diversity_threshold": 0, # Diversity threshold when greedily selecting molecules...
	},

	"true_oracle": {
		"model_name": "RFC",
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
