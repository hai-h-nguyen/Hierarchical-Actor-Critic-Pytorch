from gym.envs.registration import register

register(
    id='hac-inverted-pendulum-v0',
    entry_point='hac.domains.inverted_pendulum:InvertedPendulumEnv',
)

register(
    id='hac-mountain-car-v0',
    entry_point='hac.domains.mountain_car:MountainCarEnv',
)

# register(
#     id='hac-ur5-reacher-v0',
#     entry_point='hac.domains.ur5_reacher:Ur5Env',
#     max_episode_steps=160,
# )