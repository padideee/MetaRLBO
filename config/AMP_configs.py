

amp_000 = {
	"exp_label": "AMP-XGBoost",
	"num_proxies": 4,
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


amp_random_large = {
	"exp_label": "AMP-RANDOM-Large",
	"num_initial_samples": 250,
	"num_query_per_iter": 250,
	"policy": {
		"model_name": "RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 40,
	"mode": "val",
	"log_interval": 1,
}

amp_random_med = {
	"exp_label": "AMP-RANDOM-Medium",
	"num_initial_samples": 250,
	"num_query_per_iter": 50,
	"policy": {
		"model_name": "RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 70,
	"log_interval": 1,
}
amp_random = {
	"exp_label": "AMP-RANDOM-Medium",
	"num_initial_samples": 250,
	"num_query_per_iter": 10,
	"policy": {
		"model_name": "RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 73,
	"log_interval": 1,
}

amp_dynappo_random_large = {
	"exp_label": "AMP-DynaPPO_RANDOM-Large",
	"num_initial_samples": 250,
	"num_query_per_iter": 250,
	"policy": {
		"model_name": "DynaPPO_RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 50,
	"log_interval": 1,
}

amp_dynappo_random_med = {
	"exp_label": "AMP-DynaPPO_RANDOM-Medium",
	"num_initial_samples": 250,
	"num_query_per_iter": 50,
	"policy": {
		"model_name": "DynaPPO_RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 70,
	"log_interval": 1,
}
amp_dynappo_random = {
	"exp_label": "AMP-DynaPPO_RANDOM-Medium",
	"num_initial_samples": 250,
	"num_query_per_iter": 10,
	"policy": {
		"model_name": "DynaPPO_RANDOM",
	},
	"selection_criteria": {
		"method": "RANDOM",
	},
	"seed": 73,
	"log_interval": 1,
}




# ======================= Proxy Oracles: K Nearest Regressors


amp_knr_001 = { # Increased number of proxy models...
	"exp_label": "AMP-KNR-001",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_002 = { # Increased number of proxy models...
	"exp_label": "AMP-KNR-002",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_003 = { # Increased number of proxy models...
	"exp_label": "AMP-KNR-003",
	"num_proxies": 4, 
	"num_inner_updates": 3,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_004 = { # Increased number of proxy models...
	"exp_label": "AMP-KNR-004",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_005 = { # Increased number of proxy models...
	"exp_label": "AMP-KNR-005",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 10.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_006 = { 
	"exp_label": "AMP-KNR-006",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "true",
		"density_penalty": False,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_007 = { 
	"exp_label": "AMP-KNR-007",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_008 = { 
	"exp_label": "AMP-KNR-008",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "true",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_009 = { 
	"exp_label": "AMP-KNR-009",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_010 = { 
	"exp_label": "AMP-KNR-010",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 10.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_011 = { 
	"exp_label": "AMP-KNR-011",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_012 = { 
	"exp_label": "AMP-KNR-012",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 2, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_013 = { 
	"exp_label": "AMP-KNR-013",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 8,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_014 = { 
	"exp_label": "AMP-KNR-014",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 16,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_015 = { 
	"exp_label": "AMP-KNR-015",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_016 = { 
	"exp_label": "AMP-KNR-016",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


# ======================= Proxy Oracles: Bayesian Regression


amp_br_001 = { # Increased number of proxy models...
	"exp_label": "AMP-BR-001",
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "BR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},


	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_br_002 = { # Increased number of proxy models...
	"exp_label": "AMP-BR-002",
	"num_proxies": 4, 
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},


	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

# ======================= Proxy Oracles: Gaussian Process Regressor



# =========== Rational Quadratic
amp_gpr_rq_001 = { 
	"exp_label": "AMP-GPR_RQ-001",
	"max_num_queries": 2000,
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 16,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, 
		"config": {
			"kernel": "RationalQuadratic",
		},
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 6.0,
		}
	},
	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


# =========== Matern
amp_gpr_m_001 = { 
	"exp_label": "AMP-GPR_M-001",
	"max_num_queries": 2000,
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 16,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, 
		"config": {
			"kernel": "Matern",
		},
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 6.0,
		}
	},
	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}




# =========== RBF
amp_gpr_rbf_001 = { 
	"exp_label": "AMP-GPR_RBF-001",
	"max_num_queries": 2000,
	"num_proxies": 4, 
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4, # 
	"num_query_proxies": 16,
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 20,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 1, 
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "GPR",
		"p": 0.8, 
		"config": {
			"kernel": "RBF",
		},
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 6.0,
		}
	},
	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}






























#######################################################################################################################################

# Medium Batches -- smaller batches than DynaPPO paper but still large

amp_knr_med_001 = { 
	"exp_label": "AMP-KNR-Medium-001",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 50,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 5,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_002 = { 
	"exp_label": "AMP-KNR-Medium-002",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 50,
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

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_med_003 = { 
	"exp_label": "AMP-KNR-Medium-003",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 50,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
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

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_med_004 = { 
	"exp_label": "AMP-KNR-Medium-004",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 50,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
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

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_med_005 = { 
	"exp_label": "AMP-KNR-Medium-005",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 50,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 50,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, 
	},
	"outerloop": {
		"oracle": "proxy",
		"density_penalty": True,
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}












































##################################################################################################################################
# Large Batches -- somewhat matching the DynaPPO paper

amp_knr_large_001 = {
	"exp_label": "AMP-KNR-Large-001",
	"num_proxies": 16,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 5,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_large_002 = {
	"exp_label": "AMP-KNR-Large-002",
	"num_proxies": 16,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 5,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_large_003 = {
	"exp_label": "AMP-KNR-Large-003",
	"num_proxies": 16,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 0.5,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 5,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_large_004 = {
	"exp_label": "AMP-KNR-Large-004",
	"num_proxies": 16,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 250,
	"inner_lr": 0.1,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 5,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_005 = {
	"exp_label": "AMP-KNR-Large-005",
	"num_proxies": 16,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 10,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_large_006 = {
	"exp_label": "AMP-KNR-Large-006",
	"num_proxies": 16,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 8,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 10,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_007 = {
	"exp_label": "AMP-KNR-Large-007",
	"num_proxies": 64, # 16 -> 64
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2, # 4 -> 2
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 10,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_008 = {
	"exp_label": "AMP-KNR-Large-008",
	"num_proxies": 64, # 16 -> 64
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2, # 4 -> 2
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 20, # 10 -> 20
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_009 = {
	"exp_label": "AMP-KNR-Large-009",
	"num_proxies": 32, # 16 -> 32
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2, # 4 -> 2
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 20, # 10 -> 20
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_010 = {
	"exp_label": "AMP-KNR-Large-010",
	"num_proxies": 32, # 16 -> 32
	"num_inner_updates": 2, # 1 -> 2
	"num_meta_proxy_samples": 2, # 4 -> 2
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 20, # 10 -> 20
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_011 = {
	"exp_label": "AMP-KNR-Large-011",
	"num_proxies": 32, # 16 -> 32
	"num_inner_updates": 2, # 1 -> 2
	"num_meta_proxy_samples": 2, # 4 -> 2
	"num_initial_samples": 250,
	"num_samples_per_iter": 16,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.1,
	"num_meta_updates_per_iter": 40, # 10 -> 40
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_large_012 = {
	"exp_label": "AMP-KNR-Large-012",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 10,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_013 = {
	"exp_label": "AMP-KNR-Large-013",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 10,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_014 = {
	"exp_label": "AMP-KNR-Large-014",
	"num_proxies": 8,
	"num_inner_updates": 3,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 10,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_015 = {
	"exp_label": "AMP-KNR-Large-015",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 10,
	"entropy_reg_coeff": 0.2,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_large_016 = {
	"exp_label": "AMP-KNR-Large-016",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 300,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 1.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_017 = {
	"exp_label": "AMP-KNR-Large-017",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 300,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 10.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
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
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_018 = {
	"exp_label": "AMP-KNR-Large-018",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 300,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 100.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
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
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_019 = {
	"exp_label": "AMP-KNR-Large-019",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 300,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 100.0,
	"outer_lr": 10.0,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
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
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_020 = {
	"exp_label": "AMP-KNR-Large-020",
	"num_proxies": 8,
	"num_inner_updates": 3,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 300,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 10.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
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
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_021 = {
	"exp_label": "AMP-KNR-Large-021",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 300,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 10.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 40,
	"entropy_reg_coeff": 0.0,
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
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_022 = {
	"exp_label": "AMP-KNR-Large-022",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 300,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 10.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
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
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_023 = {
	"exp_label": "AMP-KNR-Large-023",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 300,
	"num_query_proxies": 128,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 10.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 10,
	"entropy_reg_coeff": 0.0,
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
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}



amp_knr_large_024 = {
	"exp_label": "AMP-KNR-Large-024",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 2,
	"num_initial_samples": 250,
	"num_query_proxies": 64,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 10.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 10,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 0.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_025 = {
	"exp_label": "AMP-KNR-Large-025",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 64,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 10.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 0.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_026 = {
	"exp_label": "AMP-KNR-Large-026",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 64,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}
amp_knr_large_026_1 = {
	"exp_label": "AMP-KNR-Large-026_1",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 64,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 1,
}
amp_knr_large_026_2 = {
	"exp_label": "AMP-KNR-Large-026_2",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 64,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 2,
}



amp_knr_large_027 = {
	"exp_label": "AMP-KNR-Large-027",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 64,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_large_027_1 = {
	"exp_label": "AMP-KNR-Large-027_1",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 64,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 1,
}

amp_knr_large_027_2 = {
	"exp_label": "AMP-KNR-Large-027_2",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_query_proxies": 64,
	"num_samples_per_iter": 8,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 2.0,
	"outer_lr": 0.2,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 2,
}


amp_knr_large_028 = {
	"exp_label": "AMP-KNR-Large-028",
	"num_proxies": 8,
	"num_inner_updates": 3,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}
amp_knr_large_028_1 = {
	"exp_label": "AMP-KNR-Large-028_1",
	"num_proxies": 8,
	"num_inner_updates": 3,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 1,
}
amp_knr_large_028_2 = {
	"exp_label": "AMP-KNR-Large-028_2",
	"num_proxies": 8,
	"num_inner_updates": 3,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 2,
}

amp_knr_large_029 = {
	"exp_label": "AMP-KNR-Large-029",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}
amp_knr_large_029_1 = {
	"exp_label": "AMP-KNR-Large-029_1",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 1,
}
amp_knr_large_029_2 = {
	"exp_label": "AMP-KNR-Large-029_2",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 20,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 2,
}


amp_knr_large_030 = {
	"exp_label": "AMP-KNR-Large-030",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 30,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}
amp_knr_large_030_1 = {
	"exp_label": "AMP-KNR-Large-030_1",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 30,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 1,
}
amp_knr_large_030_2 = {
	"exp_label": "AMP-KNR-Large-030_2",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 30,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 2,
}

amp_knr_large_031 = {
	"exp_label": "AMP-KNR-Large-031",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 40,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}
amp_knr_large_031_1 = {
	"exp_label": "AMP-KNR-Large-031_1",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 40,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 1,
}
amp_knr_large_031_2 = {
	"exp_label": "AMP-KNR-Large-031_2",
	"num_proxies": 8,
	"num_inner_updates": 2,
	"num_meta_proxy_samples": 8,
	"num_initial_samples": 250,
	"num_query_proxies": 32,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 250,
	"inner_lr": 5.0,
	"outer_lr": 0.5,
	"num_meta_updates_per_iter": 40,
	"entropy_reg_coeff": 0.0,
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
	"selection_criteria": { # Configs for selecting the samples
		"method": "UCB",
		"config": {
			'beta': 4.0,
		}
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 2,
}











########## start: diversity configs
amp_knr_ham_med_003 = {
	"exp_label": "AMP-KNR-Medium-003",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 100,
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 10,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
		"metric": "hamming"
	},
	"diversity": {
		"div_metric_name": "hamming",
		"div_switch": "ON"
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}


amp_knr_blast_med_003 = {
	"exp_label": "AMP-KNR-Medium-003",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 100,
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 10,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
		"metric": "hamming"
	},

	"diversity": {
		"div_metric_name": "blast",
		"div_switch": "ON"
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 1,
	"results_log_dir": "./logs",
	"seed": 73,
}

amp_knr_off_med_003 = {
	"exp_label": "AMP-KNR-Medium-003",
	"num_proxies": 8,
	"num_inner_updates": 1,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 250,
	"num_samples_per_iter": 32,
	"num_samples_per_task_update": 16,
	"num_query_per_iter": 100,
	"inner_lr": 1.0,
	"outer_lr": 1.0,
	"num_meta_updates_per_iter": 10,
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.7, # Proportion of data to sample to train proxy oracles
		"metric": "hamming"
	},

	"diversity": {
		"div_metric_name": "blast",
		"div_switch": "OFF"
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"logging": {
		"top-k": 4, # k for top-k
	},
	"log_interval": 10,
	"results_log_dir": "./logs",
	"seed": 73,
}

########## end: diversity configs

























