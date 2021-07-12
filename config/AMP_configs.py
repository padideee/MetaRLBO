

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




amp_001 = {
	"exp_label": "AMP-KNR",
	"num_proxies": 8, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 100,
	"num_samples_per_iter": 10, 
	"num_samples_per_task_update": 16, 
	"proxy_oracle": {
		"model_name": "KNR",
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


amp_002 = {
	"exp_label": "AMP-BR",
	"num_proxies": 8, 
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 100,
	"num_samples_per_iter": 10, 
	"num_samples_per_task_update": 16, 
	"proxy_oracle": {
		"model_name": "BR",
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



