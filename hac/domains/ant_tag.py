import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
from gym import spaces
from pathlib import Path
from gym.utils import seeding
import json

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'

class AntTagEnv(gym.Env):

    def __init__(self, args=None, seed=None, num_frames_skip=15, show=False):

        #################### START CONFIGS #######################

        model_name = "ant_tag_small.xml"

        if args.n_layers in [1]:
            args.time_scale = 1000
            max_actions = 1000
        elif args.n_layers in [2]:
            args.time_scale = 20
            max_actions = 400 

        self.max_actions = max_actions

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
        self.cage_max_x = 4.5
        self.cage_max_y = 4.5
        max_height = 1
        max_velo = 3

        subgoal_bounds = np.array([[-self.cage_max_x, self.cage_max_x],
                                   [-self.cage_max_y, self.cage_max_y],
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
        agent_params["subgoal_penalty"] = -args.time_scale
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

        self.extra_dim = 2 # xy coordinates of the target

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

        self.visible_radius = 4.0
        self.tag_radius = 2.0
        self.min_distance = 5.0
        self.target_step = 0.5

        self.steps_cnt = 0
        self.solved = False
        self.done = False

        self.json_writer = None
        self.list_pos = []

        self.seed(seed)

        self.max_ep_len = 200

    def full_fn(self, obs):
        return obs

    def final_selective_fn(self, obs):
        # get the final observation
        temp = obs[-self.state_dim:]

        # select the torso's position and the target's 2d position
        return np.concatenate((temp[:2], temp[-2:]))

    def final_fn(self, obs):
        return obs[-self.state_dim:]

    # everything except for the target's 2d position
    def prepare_low_obs_fn(self, obs):
        return obs[:-2]

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
    def _get_obs(self, target_pos_visible):
        if target_pos_visible:
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel, self.sim.data.mocap_pos[0][:2]))
        else:
            return np.concatenate((self.sim.data.qpos, self.sim.data.qvel, np.zeros(2)))

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

        init_position_ok = False

        while (not init_position_ok):
            # Initialize the target's position
            target_pos = self.np_random.uniform(low=[-self.cage_max_x, -self.cage_max_y], high=[self.cage_max_x, self.cage_max_y])
            ant_pos = self.np_random.uniform(low=[-self.cage_max_x, -self.cage_max_y], high=[self.cage_max_x, self.cage_max_y])

            d2target = np.linalg.norm(ant_pos - target_pos)

            if d2target > self.min_distance:
                init_position_ok = True

        self.sim.data.mocap_pos[0][:2] = target_pos

        self.sim.data.qpos[:2] = ant_pos

        # Move 2 spheres along the ant
        self.sim.data.mocap_pos[1][:2] = ant_pos
        self.sim.data.mocap_pos[2][:2] = ant_pos

        self.sim.step()

        return self._get_obs(False)

    def _move_target(self, ant_pos, current_target_pos):
        target2ant_vec = ant_pos - current_target_pos
        target2ant_vec = target2ant_vec / np.linalg.norm(target2ant_vec)

        per_vec_1 = [target2ant_vec[1], -target2ant_vec[0]]
        per_vec_2 = [-target2ant_vec[1], target2ant_vec[0]]
        opposite_vec = -target2ant_vec

        vec_list = [per_vec_1, per_vec_2, opposite_vec, np.zeros(2)]

        chosen_vec_idx = self.np_random.choice(np.arange(4), p=[0.25, 0.25, 0.25, 0.25])

        chosen_vec = np.array(vec_list[chosen_vec_idx]) * self.target_step + current_target_pos

        if abs(chosen_vec[0]) > self.cage_max_x or abs(chosen_vec[1]) > self.cage_max_y:
            chosen_vec = current_target_pos

        self.sim.data.mocap_pos[0][:2] = chosen_vec

    def _do_reveal_target(self):

        ant_pos = self.sim.data.qpos[:2]
        target_pos = self.sim.data.mocap_pos[0][:2]

        d2target = np.linalg.norm(ant_pos - target_pos)
        if (d2target < self.visible_radius):
            reveal_target_pos = True
        else:
            reveal_target_pos = False

        return reveal_target_pos

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):

        self.sim.data.ctrl[:] = action

        # TODO: Change the position of the target based on a fixed policy
        ant_pos = self.sim.data.qpos[:2]
        target_pos = self.sim.data.mocap_pos[0][:2]

        self._move_target(ant_pos, target_pos)

        # Move 2 spheres along the ant
        self.sim.data.mocap_pos[1][:2] = ant_pos
        self.sim.data.mocap_pos[2][:2] = ant_pos

        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        self.steps_cnt += 1

        ant_pos = self.sim.data.qpos[:2]

        reward = 0.0
        done = False
        env_reward = -1

        target_pos = self.sim.data.mocap_pos[0][:2]

        # + reward and terminate the episode if can tag the target
        d2target = np.linalg.norm(ant_pos - target_pos)
        if (d2target <= self.tag_radius):
            reward = 1.0
            env_reward = 0
            done = True

        self.done = done
        self.solved = (reward > 0.0)

        reveal_target_pos = self._do_reveal_target()

        return self._get_obs(reveal_target_pos), env_reward, done, {"is_success": self.solved}


    # Visualize all subgoals
    def display_subgoals(self, subgoals):
        for i in range(len(subgoals) - 1):
            self.sim.data.mocap_pos[i + 3][:3] = np.copy(subgoals[i][:3])
            self.sim.model.site_rgba[i + 3][3] = 1

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]
