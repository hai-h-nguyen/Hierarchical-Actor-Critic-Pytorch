from gym.envs.registration import register

register(
    id='car-flag-hac-v0',
    entry_point='domains.domains.car_flag:CarEnv',
    max_episode_steps=160,
)

register(
    id='bump-hac-v0',
    entry_point='hac.domains.bump:BumpEnv',
    max_episode_steps=160,
)

register(
    id='ant-reacher-hac-v0',
    entry_point='hac.domains.ant_reacher:AntEnv',
    max_episode_steps=160,
)

register(
    id='ur5-reacher-hac-v0',
    entry_point='hac.domains.ur5_reacher:Ur5Env',
    max_episode_steps=160,
)


## HAC version
register(
    id='car-flag-hac-v1',
    entry_point='hac.domains.car_flag_hac:CarEnv',
    max_episode_steps=160,
)

register(
    id='ant-reacher-hac-v1',
    entry_point='hac.domains.ant_reacher_hac:AntEnv',
    max_episode_steps=1000,
)

register(
    id='car-flag-hac-v2',
    entry_point='hac.domains.car_flag_hac_diff:CarEnv',
    max_episode_steps=160,
)