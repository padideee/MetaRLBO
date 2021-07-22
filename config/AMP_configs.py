

amp_000 = {
	"exp_label": "AMP-XGBoost",
	"num_proxies": 8, 
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


# ======================= Proxy Oracles: K Nearest Regressors

amp_knr_001 = {
	"exp_label": "AMP-KNR-001",
	"num_proxies": 8, 
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
	"log_interval": 4,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_002 = { # incr. inner updates
	"exp_label": "AMP-KNR-002",
	"num_proxies": 8, 
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
	"log_interval": 4,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_003 = {
	"exp_label": "AMP-KNR-003",
	"num_proxies": 8, 
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
	"log_interval": 4,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_004 = {
	"exp_label": "AMP-KNR-004",
	"num_proxies": 8, 
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
	"log_interval": 4,
	"results_log_dir": "./logs",
	"seed": 73,
}