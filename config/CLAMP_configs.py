clamp_knr_large_001 = {
	"exp_label": "CLAMP-KNR-Large-001",
	"task": "CLAMP-v0",
	"max_num_queries": 3000, # Maximum number of queries in experiment
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 30,
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
		"diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
	},
	"CLAMP": { # CLAMP Specific configs... do not use normally
		"true_oracle_model": "RandomForest", # RandomForest or MLP
		"data_source": "D1_target",
		"evaluation": { # post-training
			"num_query_proxies": 128,
			"num_samples_per_proxy": 500,
			"num_mols_select": 10000,
		},
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

clamp_knr_medium_001 = {
	"exp_label": "CLAMP-KNR-Medium-001",
	"task": "CLAMP-v0",
	"max_num_queries": 3000, # Maximum number of queries in experiment
	"num_proxies": 4,
	"num_inner_updates": 1,
	"num_initial_samples": 250,
	"num_query_proxies": 12,
	"num_samples_per_proxy": 75,
	"num_query_per_iter": 100,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 20,
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
		"diversity_threshold": 10, # Diversity threshold when greedily selecting molecules...
	},
	"CLAMP": { # CLAMP Specific configs... do not use normally
		"true_oracle_model": "RandomForest", # RandomForest or MLP
		"data_source": "D1_target",
		"evaluation": { # post-training
			"num_query_proxies": 128,
			"num_samples_per_proxy": 500,
			"num_mols_select": 10000,
		},
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}