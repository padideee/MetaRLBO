

DEFAULT_CONFIG = {
	"exp_label": "DEFAULT",
	"task": "AMP",
	"max_num_queries": 100000,
	"num_inner_updates": 1, 
	"num_meta_updates": 5000,
	"num_proxies": 8, 
	"num_trajs_inner_loop": 100,
	"num_meta_proxy_samples": 10,
	"num_initial_samples": 100,
	"num_samples_per_iter": 5, 
	"num_samples_per_task_update": 16, 
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"policy": {
		"model_name": "MLP",
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 5, # k for top-k
	},
	"log_interval": 25,
	"results_log_dir": "./logs",
	"seed": 73,
}