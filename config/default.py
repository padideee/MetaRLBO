

DEFAULT_CONFIG = {
	"exp_label": "DEFAULT",
	"task": "AMP",
	"max_num_queries": 100000,
	"num_inner_updates": 1, # Number of inner loop updates per meta update
	"num_meta_updates": 5000,
	"num_proxies": 8, # Number of proxies (i.e., tasks)
	"num_meta_proxy_samples": 10, # Number of samples (per proxy) to perform meta updates with
	"num_initial_samples": 100, # Number of initial samples to train proxy models on
	"num_samples_per_iter": 5, # Number of samples per proxy/task to query the true oracle on (after inner loop updates)
	"num_samples_per_task_update": 16, # Number of samples to train policy on per proxy
	"inner_lr": 1e-1,
	"outer_lr": 1e-3,
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
		"top-k": 5, # k for top-k  (per proxy) --- This needs to be lower than "num_samples_per_iter"
	},
	"log_interval": 25,
	"results_log_dir": "./logs",
	"seed": 73,
}