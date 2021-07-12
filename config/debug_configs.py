


debug = {
	"exp_label": "DEBUG",
	"num_proxies": 2, 
	"num_inner_updates": 3, 
	"num_trajs_inner_loop": 10,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 10,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 4, 
	"proxy_oracle": {
		"model_name": "BR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"log_interval": 1,
}


debug_BR = {
	"exp_label": "DEBUG-BR",
	"num_proxies": 2, 
	"num_inner_updates": 3, 
	"num_trajs_inner_loop": 10,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 10,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 4, 
	"proxy_oracle": {
		"model_name": "BR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"log_interval": 1,
}



debug_RR = {
	"exp_label": "DEBUG-RR",
	"num_proxies": 2, 
	"num_inner_updates": 3, 
	"num_trajs_inner_loop": 10,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 10,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 4, 
	"proxy_oracle": {
		"model_name": "BR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"log_interval": 1,
}



debug_KNR = {
	"exp_label": "DEBUG-KNR",
	"num_proxies": 2, 
	"num_inner_updates": 3, 
	"num_trajs_inner_loop": 10,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 20,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 4, 
	"proxy_oracle": {
		"model_name": "KNR",
		"p": 0.8, # Proportion of data to sample to train proxy oracles
	},
	"log_interval": 1,
}

