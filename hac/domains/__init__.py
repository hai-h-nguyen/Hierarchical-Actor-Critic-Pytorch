from gym.envs.registration import register

register(
    id='hac-inverted-pendulum-v0',
    entry_point='hac.domains.inverted_pendulum:InvertedPendulumEnv',
    max_episode_steps=160,
)

register(
    id='hac-ur5-reacher-v0',
    entry_point='hac.domains.ur5_reacher:Ur5Env',
    max_episode_steps=160,
)