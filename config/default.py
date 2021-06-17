

DEFAULT_CONFIGS = {
	"task": "AMP",
	"max_num_queries": 100000,
	"num_meta_updates": 5000,
	"num_proxies": 8, 
	"num_trajs_inner_loop": 100,
	"num_meta_proxy_samples": 10,
	"num_initial_samples": 100,
	"num_samples_per_iter": 5, 
	"num_samples_per_task_update": 16, 
	"proxy_oracle": {
		"model_name": "RFC",
	},

	"true_oracle": {
		"model_name": "RFC",
	},
	"policy": {
		"hidden_dim": 100,
	}
}