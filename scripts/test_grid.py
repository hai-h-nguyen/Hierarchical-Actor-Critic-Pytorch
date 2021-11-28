import gym

from hac.domains import *

env = gym.make('hierq-grid-world-v0')
env.reset()

for i in range(100):
    action = env.action_space.sample()
    print(action)
    env.step(action)