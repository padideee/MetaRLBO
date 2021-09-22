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