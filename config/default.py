

DEFAULT_CONFIG = {
	"exp_label": "DEFAULT",
	"task": "AMP",
	"max_num_queries": 5000,
	"query_storage_size": 100000,
	"num_inner_updates": 1, # Number of inner loop updates per meta update
	"num_meta_updates": 5000,
	"num_proxies": 8, # Number of proxies (i.e., tasks)
	"num_query_proxies": 64, # Number of proxies used to generate molecules to query from...
	"num_meta_proxy_samples": 10, # Number of samples (per proxy) to perform meta updates with
	"num_initial_samples": 100, # Number of initial samples to train proxy models on

	"num_samples_per_iter": 5, # Number of samples per proxy/task to (select from) to query the true oracle 
	"num_query_per_iter": 8,  # Number of samples to query the true oracle (per iteration)
	"num_samples_per_task_update": 16, # Number of samples to train policy on per proxy
	"num_meta_updates_per_iter": 1,
	"inner_lr": 1e-1,
	"outer_lr": 1e-2,
	"outerloop_oracle": "proxy",  # Options: proxy, true
	"proxy_oracle": {
		"model_name": "KNR", 
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"policy": {
		"model_name": "MLP",
	},


	"select_samples": { # Configs for selecting the samples
		"method": "PROXY_MEAN", 
	},
	"true_oracle": {
		"model_name": "RFC",
		"config": {
			"n_estimators": 128,
		}
	},
	"logging": {
		"top-k": 5, # k for top-k  (per proxy) --- This needs to be lower than "num_samples_per_iter"
	},
	"env": {
		"lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2,
	},
	"data_source": "DynaPPO", # Either: DynaPPO or Custom (Custom being data Padideh generated)
	"mode": "test", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 25,
	"results_log_dir": "./logs",
	"seed": 73,
}