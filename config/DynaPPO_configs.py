dynappo_debug = {
    "exp_label": "dynappo_debug",
    "task": "AMP-v0",
    "max_num_queries": 3000, # Maximum number of queries in experiment
    "query_storage_size": 100000, # Maximum number of queries allowed in the storage
    "num_updates_per_iter": 72,
    "ppo_config": { # Leo: This should be merged into train_policy_config --
        "clip_param": 0.2,
        "ppo_epoch": 4,
        "num_mini_batch": 4,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "lr": 7e-4,
        "eps": 1e-5,
        "max_grad_norm": 0.5,
        "use_gae": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "use_proper_time_limits": False,
        # "num_steps": 10,
    },
    "proxy_oracle": {
        "p": 1.0, # Proportion of data to sample to train proxy oracles -- Fixed to 1.0 for DynaPPO!
    },
    "policy": {
        "num_steps": 150,
    }
    "save_interval": 10, # Save model every n batch queries
    "num_processes": 8, 
    "results_log_dir": "./logs", 
    "seed": 73,
}