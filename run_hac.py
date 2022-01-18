"""
This is the starting file for the Hierarchical Actor-Critc (HAC) algorithm.  The below script processes the command-line options specified
by the user and instantiates the environment and agent. 
"""

from hac.hac.options import parse_options
from hac.hac.agent import Agent
from hac.hac.run_HAC import run_HAC, test_HAC
import gym
import logging
import sys
import os
from hac.domains import *
from hac.utils import NoisyObsWrapper, RandomActWrapper

# Determine training options specified by user.  The full list of available options can be found in "options.py" file.
args = parse_options()
logging.basicConfig(stream=sys.stdout, level=args.log_level)

LOG_DIR_PREFIX = "results/logs/hac"
log_dir = os.path.join(LOG_DIR_PREFIX, args.env, str(args.n_layers)+"-levels", str(args.seed))
os.makedirs(log_dir, exist_ok=True)

# Instantiate the agent and Mujoco environment.
env = gym.make(args.env, args=args, seed=args.seed, show=args.show)

if args.noisy_obs:
    print("Add noise to the obs: ")
    env = NoisyObsWrapper(env)

if args.random_act:
    print("Random action: ")
    env = RandomActWrapper(env)   

agent = Agent(args, env, log_dir)

# Begin training
if args.test:
    test_HAC(args, env, agent)
else:
    run_HAC(args, env, agent)