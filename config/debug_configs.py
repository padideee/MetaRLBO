



debug = {
	"exp_label": "DEBUG",
	"num_proxies": 2, 
	"num_inner_updates": 3, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 10,
	"num_samples_per_iter": 4, 
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

# Varying Proxy Oracle Models
debug_BR = {
	"exp_label": "DEBUG-BR",
	"num_proxies": 2, 
	"num_inner_updates": 3, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 10,
	"num_samples_per_iter": 4, 
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
	"num_samples_per_iter": 4, 
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
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 20,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 4, 
	"inner_lr": 1e0,
	"outer_lr": 1e-1,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 1,
}


debug_KNR_OL = {
	"exp_label": "DEBUG-KNR",
	"num_proxies": 2, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 20,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 4, 
	"inner_lr": 1e0,
	"outer_lr": 1e-1,
	"outerloop_oracle": "true",
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"env": {
		"lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"log_interval": 1,
}




debug_mid = {
	"exp_label": "DEBUG-Medium",
	"num_proxies": 8, 
	"num_inner_updates": 1, 
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16, 
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
	"num_samples_per_iter": 4, 
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
	"num_samples_per_iter": 4,
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