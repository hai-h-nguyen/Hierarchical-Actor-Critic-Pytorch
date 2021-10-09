from gym.envs.registration import register

register(
    id='ur5-reacher-v0',
    entry_point='hac.domains.ur5_reacher:Ur5Env',
    max_episode_steps=160,
)