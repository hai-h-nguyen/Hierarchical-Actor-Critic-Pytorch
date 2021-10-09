import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
from gym import spaces
from pathlib import Path
from gym.utils import seeding

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'

class AntEnv(gym.Env):

    def __init__(self, obs_type='coodinate', args=None, seed=None, num_frames_skip=15, show=False):

        #################### START CONFIGS #######################
        if args is not None:
            if args.n_layers in [1]:
                args.time_scale = 100
                max_actions = 1000
            elif args.n_layers in [2]:
                args.time_scale = 25
                max_actions = 600
            elif args.n_layers in [3]:
                args.time_scale = 10
                max_actions = 600

        self.max_actions = max_actions

        self.prepare_high_obs_fn = lambda state: state



        num_frames_skip = num_frames_skip

        model_name = "ant_reacher.xml"

        initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
        initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
        initial_joint_ranges[0] = np.array([-6,6])
        initial_joint_ranges[1] = np.array([-6,6])

        initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges)-1,2))),0)

        # functions to project state to goal
        self.project_state_to_subgoal = lambda sim, state: \
                                    np.concatenate((sim.data.qpos[:2], np.array([1 if sim.data.qpos[2] > 1 else sim.data.qpos[2]]), 
                                    np.array([3 if sim.data.qvel[i] > 3 else -3 if sim.data.qvel[i] < -3 else sim.data.qvel[i] for i in range(2)])))

        self.project_state_to_endgoal = lambda sim, state: state[:3]

        # The subgoal space in the Ant Reacher task is the desired (x,y,z) position and (x,y,z) translational velocity of the torso
        cage_max_dim = 7.5
        max_height = 1
        max_velo = 3
        subgoal_bounds = np.array([[-cage_max_dim,cage_max_dim],
                                   [-cage_max_dim,cage_max_dim],
                                   [0,max_height],
                                   [-max_velo, max_velo],
                                   [-max_velo, max_velo]])

        len_threshold = 0.4
        height_threshold = 0.5

        self.goal_space_train = [[-cage_max_dim,cage_max_dim],[-cage_max_dim,cage_max_dim],[0, max_height]]
        self.goal_space_test = [[-cage_max_dim,cage_max_dim],[-cage_max_dim,cage_max_dim],[0, max_height]]

        # Set subgoal achievement thresholds
        velo_threshold = 0.8
        subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, velo_threshold, velo_threshold])

        # Configs for agent
        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3
        agent_params["random_action_perc"] = 0.3

        agent_params["atomic_noise"] = [0.2 for i in range(8)]
        agent_params["subgoal_noise"] = [0.2 for i in range(len(subgoal_thresholds))]

        agent_params["num_pre_training_episodes"] = 30

        agent_params["episodes_to_store"] = 500
        agent_params["num_exploration_episodes"] = 100

        #################### END CONFIGS #######################
        
        self.agent_params = agent_params

        self.name = model_name

        MODEL_PATH = ASSETS_PATH / self.name

        # Create Mujoco Simulation
        self.model = load_model_from_path(str(MODEL_PATH))
        self.sim = MjSim(self.model)

        # choose what type of observation that the priest will tell the agent: coordinate - the location of the heaven, 
        # something else: direction (left/right) only
        self.obs_type = obs_type

        if self.obs_type in ['coodinate']:
            self.extra_dim = 2 # Going to the priest will tell (x, y) coordinate of the heaven
        else:
            self.extra_dim = 1 # Going to the priest will tell the direction to the heaven (left/right)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel) + self.extra_dim # State will include (i) joint angles and (ii) joint velocities, extra info
        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds = self.sim.model.actuator_ctrlrange[:,1] # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # End goal/subgoal thresholds
        self.subgoal_thresholds = subgoal_thresholds
        len_threshold = 0.4
        height_threshold = 0.5
        self.endgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold])

        if args is not None:
            agent_params["subgoal_penalty"] = -args.time_scale

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

        self.endgoal_dim = 3

        # For Gym interface
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32            
        )

        self.heaven_hell = [[-6.25, 6.75], [6.25, 6.75]]
        self.priest_pos = [5.25, -5.75]
        self.radius = 1.9

        self.steps_cnt = 0
        self.solved = False
        self.done = False

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
    def get_state(self, reveal_heave_pos, at_reset=False):
        if self.obs_type in ['coodinate']:
            if at_reset:
                heaven_pos = np.zeros_like(self.heaven_pos)
                return np.concatenate((self.sim.data.qpos, self.sim.data.qvel, heaven_pos))

            heaven_pos = self.heaven_pos if reveal_heave_pos else np.zeros_like(self.heaven_pos)
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel, heaven_pos))

        else:
            if at_reset:
                return np.concatenate((self.sim.data.qpos, self.sim.data.qvel, np.zeros(1)))

            heaven_direction = np.sign(self.heaven_pos[0])
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel, np.array([heaven_direction])))

    # Reset simulation to state within initial state specified by user
    def reset(self):

        self.steps_cnt = 0
        self.done = False
        self.solved = False

        # Reset controls
        self.sim.data.ctrl[:] = 0

        coin_face = self.np_random.rand() >= 0.5
        self.heaven_pos = self.heaven_hell[coin_face]
        self.hell_pos = self.heaven_hell[not coin_face]

        # Changing the color of heaven/hell areas
        if coin_face:
            self.sim.model.site_rgba[2] = [0,1,0,0.5]
            self.sim.model.site_rgba[4] = [1,0,0,0.5]
        else:
            self.sim.model.site_rgba[4] = [0,1,0,0.5]
            self.sim.model.site_rgba[2] = [1,0,0,0.5]            


        # Set initial joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

        # Initialize ant's position
        self.sim.data.qpos[0] = 0.0
        self.sim.data.qpos[1] = 0.0

        self.sim.step()

        # Return state
        return self.get_state(False, at_reset=True)

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):

        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        self.steps_cnt += 1

        ant_pos = self.sim.data.qpos[:2]

        d2heaven = np.linalg.norm(ant_pos - self.heaven_pos)

        reward = 0.0
        if (d2heaven < self.radius):
            reward = 1.0

        d2priest = np.linalg.norm(ant_pos - self.priest_pos)
        if (d2priest < self.radius):
            reveal_heaven_pos = True
        else:
            reveal_heaven_pos = False

        self.done = (reward > 0.0)
        self.solved = (reward > 0.0)

        return self.get_state(reveal_heaven_pos), reward, False, {}


    # Visualize all subgoals
    def display_subgoals(self, subgoals):
        for i in range(len(subgoals) - 1):
            self.sim.data.mocap_pos[i + 1][:3] = np.copy(subgoals[i][:3])

    def display_endgoal(self, endgoal):
        self.sim.data.mocap_pos[0][:3] = np.copy(endgoal[:3])

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]