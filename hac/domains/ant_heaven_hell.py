import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
from gym import spaces
from pathlib import Path
from gym.utils import seeding
import json

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'

GREEN = [0, 1, 0, 0.5]
RED = [1, 0, 0, 0.5]

class AntEnv(gym.Env):

    def __init__(self, args=None, seed=None, num_frames_skip=15, show=False):

        #################### START CONFIGS #######################

        model_name = "ant_heaven_hell.xml"

        if args.n_layers in [1]:
            args.time_scale = 1000
            max_actions = 1000
        elif args.n_layers in [2]:
            args.time_scale = 20
            max_actions = 400        

        initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
        initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
        initial_joint_ranges[0] = np.array([-6,6])
        initial_joint_ranges[1] = np.array([-6,6])

        initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges)-1,2))),0)

        # functions to project state to goal
        project_state_to_subgoal = lambda sim, state: \
                                    np.concatenate((sim.data.qpos[:2], np.array([1 if sim.data.qpos[2] > 1 else sim.data.qpos[2]]), 
                                    np.array([3 if sim.data.qvel[i] > 3 else -3 if sim.data.qvel[i] < -3 else sim.data.qvel[i] for i in range(2)])))

        # The subgoal space in the Ant Reacher task is the desired (x,y,z) position and (x,y,z) translational velocity of the torso
        cage_max_x = 7.5
        cage_max_y_low = -1.25
        cage_max_y_high = 7.5
        max_height = 1
        max_velo = 3

        # TODO:
        subgoal_bounds = np.array([[-cage_max_x, cage_max_x],
                                   [cage_max_y_low, cage_max_y_high],
                                   [0, max_height],
                                   [-max_velo, max_velo],
                                   [-max_velo, max_velo]])

        len_threshold = 0.4
        height_threshold = 0.4

        # Set subgoal achievement thresholds
        velo_threshold = 0.8
        subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, velo_threshold, velo_threshold])

        # Configs for agent
        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3
        agent_params["random_action_perc"] = 0.3

        agent_params["subgoal_penalty"] = -args.time_scale

        agent_params["atomic_noise"] = [0.2 for i in range(8)]
        agent_params["subgoal_noise"] = [0.2 for i in range(len(subgoal_thresholds))]

        agent_params["num_pre_training_episodes"] = 30

        agent_params["episodes_to_store"] = 500
        agent_params["num_exploration_episodes"] = 100

        #################### END CONFIGS #######################
        
        self.agent_params = agent_params

        self.name = model_name
        self.max_actions = max_actions

        MODEL_PATH = ASSETS_PATH / self.name

        # Create Mujoco Simulation
        self.model = load_model_from_path(str(MODEL_PATH))
        self.sim = MjSim(self.model)

        self.extra_dim = 1 # Going to the priest will tell the direction only (left/right)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel) + self.extra_dim # State will include (i) joint angles and (ii) joint velocities, extra info
        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds = self.sim.model.actuator_ctrlrange[:,1] # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        # Projection functions
        self.project_state_to_subgoal = project_state_to_subgoal

        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        # End goal/subgoal thresholds
        self.subgoal_thresholds = subgoal_thresholds

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

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

        self.heaven_hell = [[-6.25, 6.0], [6.25, 6.0]]
        self.priest_pos = [0.0, 6.0]
        self.radius = 2.0

        self.steps_cnt = 0
        self.solved = False
        self.done = False

        self.left_id = self.model.site_name2id('left_area')
        self.right_id = self.model.site_name2id('right_area')

        self.json_writer = None
        self.list_pos = []

        self.seed(seed)

        self.max_ep_len = 200

    def full_fn(self, obs):
        return obs

    def final_selective_fn(self, obs):
        # get the final observation
        temp = obs[-self.state_dim:]

        # select the torso's position and the direction bit
        return np.concatenate((temp[:2], temp[-1:]))

    def final_fn(self, obs):
        return obs[-self.state_dim:]

    def prepare_low_obs_fn(self, obs):
        return obs[:-1]

    def save_position(self, total_env_steps):
        _qpos = self.sim.data.qpos
        self.list_pos.append([_qpos[0], _qpos[1]])

        if total_env_steps % 1000_000 == 0 and total_env_steps > 0:
            print("Saving positions at: ", total_env_steps)
            self.json_writer = open('rhac_' + str(total_env_steps) + '.txt', 'w')
            json.dump(list(self.list_pos), self.json_writer)
            self.json_writer.close()
            self.json_writer = None

    # Get state, which concatenates joint positions and velocities
    def _get_obs(self, reveal_heaven_direction, at_reset=False):
        if at_reset:
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel, np.zeros(1)))

        heaven_direction = self.heaven_direction * np.ones(1) if reveal_heaven_direction else np.zeros(1)

        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel, heaven_direction))

    # Reset simulation to state within initial state specified by user
    def reset(self):

        self.steps_cnt = 0
        self.done = False
        self.solved = False

        # Reset controls
        self.sim.data.ctrl[:] = 0

        # Set initial joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = self.np_random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = self.np_random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

        # Initialize ant's position
        self.sim.data.qpos[0] = self.np_random.uniform(-1.0, 1.0)
        self.sim.data.qpos[1] = self.np_random.uniform(0.0, 1.0)

        # Randomize the side of heaven
        coin_face = self.np_random.rand() >= 0.5

        # -1: heaven on left, 1: heaven on the right
        self.heaven_pos = self.heaven_hell[coin_face]
        self.hell_pos = self.heaven_hell[not coin_face]

        self.heaven_direction = np.sign(self.heaven_pos[0])
        
        # Changing the color of heaven/hell areas
        if self.heaven_direction > 0:

            # print("Heaven on the right")

            # heaven on the right 
            self.sim.model.site_rgba[self.right_id] = GREEN
            self.sim.model.site_rgba[self.left_id] = RED
        else:
            # print("Heaven on the left")

            # heaven on the left
            self.sim.model.site_rgba[self.left_id] = GREEN
            self.sim.model.site_rgba[self.right_id] = RED
            
        self.sim.step()

        # Return state
        return self._get_obs(False, at_reset=True)

    def _do_reveal_target(self):

        ant_pos = self.sim.data.qpos[:2]

        d2priest = np.linalg.norm(ant_pos - self.priest_pos)
        if (d2priest < self.radius):
            reveal_heaven_direction = True
        else:
            reveal_heaven_direction = False

        return reveal_heaven_direction

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
        done = False
        env_reward = -1

        # + reward and terminate the episode if going to heaven
        if (d2heaven <= self.radius):
            reward = 1.0
            env_reward = 0
            done = True

        d2hell = np.linalg.norm(ant_pos - self.hell_pos)

        # terminate the episode if going to  hell
        if (d2hell <= self.radius):
            env_reward = self.steps_cnt - self.max_ep_len
            done = True

        self.done = done
        self.solved = (reward > 0.0)

        reveal_heaven_direction = self._do_reveal_target()

        return self._get_obs(reveal_heaven_direction), env_reward, done, {"is_success": self.solved}


    # Visualize all subgoals
    def display_subgoals(self, subgoals):
        for i in range(len(subgoals) - 1):
            self.sim.data.mocap_pos[i + 3][:3] = np.copy(subgoals[i][:3])
            self.sim.model.site_rgba[7 + i][3] = 1 # site index for subgoals starting at 7

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]
