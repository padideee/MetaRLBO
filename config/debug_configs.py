

debug = {
	"num_proxies": 2, 
	"num_inner_updates": 1, 
	"num_trajs_inner_loop": 10,
	"num_meta_proxy_samples": 4,
	"num_initial_samples": 10,
	"num_samples_per_iter": 4, 
	"num_samples_per_task_update": 4, 
	"log_interval": 1,
	"proxy_oracle": {
		"model_name": "RR",
		"p": 0.9, # Proportion of data to sample to train proxy oracles
	},
}

