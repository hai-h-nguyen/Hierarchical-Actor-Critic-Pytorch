from gym.envs.registration import register

register(
    id='hac-inverted-pendulum-v0',
    entry_point='hac.domains.inverted_pendulum:InvertedPendulumEnv',
)

register(
    id='hac-mountain-car-v0',
    entry_point='hac.domains.mountain_car:MountainCarEnv',
)

register(
    id='hac-ur5-reacher-v0',
    entry_point='hac.domains.ur5_reacher:UR5Env',
)

register(
    id='hac-ant-four-rooms-v0',
    entry_point='hac.domains.ant_four_rooms:AntFourRoomsEnv',
)

register(
    id='hac-ant-reacher-v0',
    entry_point='hac.domains.ant_reacher:AntReacherEnv',
)

register(
    id='hac-peg-in-hole-v0',
    entry_point='hac.domains.fetch.reach:FetchReachEnv',
)