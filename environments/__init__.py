from gym.envs.registration import register

# AMP
# ----------------------------------------

register( # This is an issue...
    'AMP-v0',
    entry_point='environments.AMP_env:AMPEnv',
    kwargs={},
)
register( # This is an issue...
    'CLAMP-v0',
    entry_point='environments.AMP_env:AMPEnv',
    kwargs={},
)



register( # This is an issue...
    'AltIsing20-v0',
    entry_point='environments.AltIsing_env:AltIsingEnv',
    kwargs={'max_length': 20, 'vocab_size': 20},
)

register( # This is an issue...
    'AltIsing50-v0',
    entry_point='environments.AltIsing_env:AltIsingEnv',
    kwargs={'max_length': 50, 'vocab_size': 20},
)

register( # This is an issue...
    'AltIsing100-v0',
    entry_point='environments.AltIsing_env:AltIsingEnv',
    kwargs={'max_length': 100, 'vocab_size': 20},
)


register( # This is an issue...
    'RNA14-v0',
    entry_point='environments.RNA_env:RNAEnv',
    kwargs={'max_length': 14, 'vocab_size': 4},
)



