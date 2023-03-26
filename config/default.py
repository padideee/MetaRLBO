

DEFAULT_CONFIG = {
	"exp_label": "DEFAULT",
	"use_metalearner": True,
	"use_ensemble_learner": False,
	"use_rl_learner": False,
	"task": "AMP-v0",
	"max_num_queries": 3000, # Maximum number of queries in experiment 
	"query_storage_size": 100000, # Maximum number of queries allowed in the storage
	"num_inner_updates": 1, # Number of inner loop updates
	"num_meta_updates": 20000, # Max. # of meta updates (ignore)
	"num_proxies": 4, # Number of proxies for training (i.e., tasks)
	"num_query_proxies": 4, # Number of proxies used to generate molecules to query from...
	"num_initial_samples": 250, # Number of initial samples to train proxy models on (generated via Random Policy)
	"num_samples_per_proxy": 16, # Number of proposed samples per proxy/task to select from to query the true oracle 
	"num_query_per_iter": 20,  # Number of samples to query the true oracle (per iteration) -- (i.e. equiv. to batch size for Bayesian Opt.)
	"num_meta_updates_per_iter": 1, # Number of meta updates before each query to the true oracle
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"outerloop": {
		"oracle": "proxy", # Options: "true" or "proxy"
		"density_penalty": True, # Options: True or False
	},
	"proxy_oracle": {
		"model_name": "KNR", # Options: KNR, GPR, XGB, BR, RFR, etc... (see oracles/models.py)
		"UQ" : "MCdropout",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
		"metric": "minkowski" 
	},
	"policy": {
		"model_name": "MLP",
		"num_steps": 150, # number of steps (per env) before updating... ensure this is at least as big as the length of the episode of the environment
		"num_meta_steps": 50,
	},
	"exp_policy": { #TODO: change the size for AMP env and Ising env
		"model_name": "MLP",
		"input_size": 56,  # 14*4
		"hidden_size": 28,  # 14 * 2
		"output_size": 14,
	},
	"diversity": {
		"div_metric_name": "hamming", # Options: "hamming" or "blast" (Note: blast is slow)
		"div_switch": "ON" # switches the diversity bonus ON / OFF -- (Note: there's overlap with ["outerloop"]["density_penalty"]... be careful)
	},
	"reward" : "E+IN", # "E+IN" or "IN": pure exploration
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB", 
		"config": {
			'beta': 4.0,  # score = mean + beta * std
		},
		"diversity_threshold": 1, # Diversity threshold when greedily selecting molecules...
	},
	"true_oracle": { # Typically, do not change this: (Same configs as DynaPPO)
		"model_name": "RFC",
		"config": {
			"n_estimators": 128,
		}
	},
	"metalearner": {
		"method": "REINFORCE",  # REINFORCE or TRPO
	},
	"logging": { # Deprecated (Ignore)
		"top-k": 5, # k for top-k  (per proxy) --- This needs to be lower than "num_samples_per_proxy"
	},
	"env": { # See DynaPPO paper for these configs
		"lambda": 0.1, # Diversity hyperparameter -- higher is more penalty for more similar mols.
		"radius": 2, 
	},
	"CLAMP": { # CLAMP Specific configs... do not use normally
		"true_oracle_model": "RandomForest", # RandomForest or MLP
		"data_source": "D1_target",
		"evaluation": { # post-training
			"num_query_proxies": 64,
			"num_samples_per_proxy": 500,
			"num_mols_select": 10000,
			"actual_model": "MLP",
		},
	},
	"use_baseline": False,
	"reset_policy_per_round": False,
	"reset_exp_policy_per_round": True,
	"query_reward_in_env": False, # Faster to do the querying outside of the env since we can do it in batches.
	"entropy_reg_coeff": 0.0, # Deprecated (Ignore)
	"data_source": "DynaPPO", # Either: DynaPPO or Custom (Custom being data Padideh generated)
	"mode": "test", # mode -- val (hyperparameter opt.), test (eval. )
	"log_interval": 1, 
	"save_interval": 10, # Save model every n batch queries
	"num_processes": 8, 
	"results_log_dir": "./logs", 
	"seed": 73,
}