from gym.envs.registration import register

register(
    id='hierq-grid-world-v0',
    entry_point='hac.domains.grid_world:Grid_World',
)
