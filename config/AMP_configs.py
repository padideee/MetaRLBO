

amp_000 = {
	"exp_label": "AMP",
	"num_inner_updates": 1, 
	"num_meta_updates": 5000,
	"num_proxies": 8, 
	"num_meta_proxy_samples": 10,
	"num_initial_samples": 100,
	"num_samples_per_iter": 5, 
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
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}