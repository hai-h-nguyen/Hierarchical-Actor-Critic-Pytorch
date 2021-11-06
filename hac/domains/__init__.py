from gym.envs.registration import register

register(
    id='hac-car-flag-v0',
    entry_point='hac.domains.car_flag:CarEnv',
)

register(
    id='hac-two-boxes-check-v0',
    entry_point='hac.domains.two_boxes_check:BoxEnv',
)

register(
    id='hac-ant-heaven-hell-v0',
    entry_point='hac.domains.ant_heaven_hell:AntEnv',
)

register(
    id='hac-ant-tag-v0',
    entry_point='hac.domains.ant_tag:AntTagEnv',
)