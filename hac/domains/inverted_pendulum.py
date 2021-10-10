import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
from gym import spaces
from pathlib import Path
from gym.utils import seeding

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'

def bound_angle(angle):
    bounded_angle = angle % (2*np.pi)

    if np.absolute(bounded_angle) > np.pi:
        bounded_angle = -(np.pi - bounded_angle % np.pi)

    return bounded_angle

class InvertedPendulumEnv(gym.Env):

    def __init__(self, args=None, seed=None, num_frames_skip=10, show=False):

        #################### START CONFIGS #######################
        if args is not None:
            if args.n_layers in [1]:
                args.time_scale = 1000
                max_actions = 1000
            elif args.n_layers in [2]:
                args.time_scale = 32
                max_actions = 1000
            elif args.n_layers in [3]:
                args.time_scale = 10
                max_actions = 1000

        self.max_actions = max_actions

        model_name = "pendulum.xml"

        initial_state_space = np.array([[np.pi/4, 7*np.pi/4],[-0.05,0.05]])

        self.goal_space_train = [[np.deg2rad(-16),np.deg2rad(16)],[-0.6,0.6]]
        self.goal_space_test = [[0,0],[0,0]]

        # functions to project state to goal
        self.project_state_to_endgoal = lambda sim, state: np.array([bound_angle(sim.data.qpos[0]), 15 if state[2] > 15 else -15 if state[2] < -15 else state[2]])
        self.project_state_to_subgoal = lambda sim, state: np.array([bound_angle(sim.data.qpos[0]), 15 if state[2] > 15 else -15 if state[2] < -15 else state[2]])

        endgoal_thresholds = np.array([np.deg2rad(9.5), 0.6])

        subgoal_bounds = np.array([[-np.pi,np.pi],[-15,15]])

        # Configs for agent
        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3
        agent_params["random_action_perc"] = 0.3

        agent_params["atomic_noise"] = [0.1 for i in range(1)]
        agent_params["subgoal_noise"] = [0.1 for i in range(2)]

        agent_params["num_pre_training_episodes"] = -1
        agent_params["episodes_to_store"] = 200
        agent_params["num_exploration_episodes"] = 50

        #################### END CONFIGS #######################
        
        self.agent_params = agent_params

        self.name = model_name

        MODEL_PATH = ASSETS_PATH / self.name

        # Create Mujoco Simulation
        self.model = load_model_from_path(str(MODEL_PATH))
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.state_dim = 2*len(self.sim.data.qpos) + len(self.sim.data.qvel)
        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds = self.sim.model.actuator_ctrlrange[:,1] # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.end_goal_dim = len(self.goal_space_test)
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # End goal/subgoal thresholds
        self.subgoal_thresholds = np.array([np.deg2rad(9.5), 0.6])
        self.endgoal_thresholds = np.array([np.deg2rad(9.5), 0.6])

        if args is not None:
            agent_params["subgoal_penalty"] = -args.time_scale

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

        self.endgoal_dim = len(self.goal_space_test)

        self.seed(seed)

    def get_next_goal(self, test):
        end_goal = np.zeros((len(self.goal_space_test)))

        if not test:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_train[i][0], self.goal_space_train[i][1])
        else:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_test[i][0], self.goal_space_test[i][1])

        self.display_endgoal(end_goal)

        return end_goal  

    # Get state, which concatenates joint positions and velocities
    def get_state(self):
        return np.concatenate([np.cos(self.sim.data.qpos),np.sin(self.sim.data.qpos),
                    self.sim.data.qvel])

    # Reset simulation to state within initial state specified by user
    def reset(self):

        self.steps_cnt = 0
        self.done = False
        self.solved = False

        # Reset controls
        self.sim.data.ctrl[:] = 0

        # Set initial joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

        self.sim.step()

        # Return state
        return self.get_state()

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):

        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        return self.get_state(), 0, False, {}


    # Visualize all subgoals
    def display_subgoals(self, subgoals):
        for i in range(1, len(subgoals)):
            self.sim.data.mocap_pos[i] = np.array([0.5*np.sin(subgoals[i-1][0]),0,0.5*np.cos(subgoals[i-1][0])+0.6])
            self.sim.model.site_rgba[i][3] = 1

    def display_endgoal(self, endgoal):
        self.sim.data.mocap_pos[0] = np.array([0.5*np.sin(endgoal[0]),0,0.5*np.cos(endgoal[0])+0.6])

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]