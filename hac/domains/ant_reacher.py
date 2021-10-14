import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
from gym import spaces
from pathlib import Path
from gym.utils import seeding

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'


class AntReacherEnv(gym.Env):

    def __init__(self, args, seed, max_actions=1200, num_frames_skip=10, show=False):

        #################### START CONFIGS #######################
        # TODO: Remove 1 layer case?
        if args.n_layers in [1]:
            args.time_scale = 1000
            max_actions = 1000
        elif args.n_layers in [2]:
            args.time_scale = 23
            max_actions = 500
        elif args.n_layers in [3]:
            args.time_scale = 10
            max_actions = 500

        timesteps_per_action = 15
        num_frames_skip = timesteps_per_action

        model_name = "ant_reacher.xml"

        initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
        initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
        initial_joint_ranges[0] = np.array([-9.5,9.5])
        initial_joint_ranges[1] = np.array([-9.5,9.5])

        initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges)-1,2))),0)

        # TODO: remove these
        max_range = 9.5
        goal_space_train = [[-max_range,max_range],[-max_range,max_range],[0.45,0.55]]
        goal_space_test = [[-max_range,max_range],[-max_range,max_range],[0.45,0.55]]

        # functions to project state to goal
        project_state_to_endgoal = lambda sim, state: state[:3]
        project_state_to_subgoal = lambda sim, state: np.concatenate((sim.data.qpos[:2], np.array([1 if sim.data.qpos[2] > 1 else sim.data.qpos[2]]), 
                                    np.array([3 if sim.data.qvel[i] > 3 else -3 if sim.data.qvel[i] < -3 else sim.data.qvel[i] for i in range(2)])))

        # The subgoal space in the Ant Reacher task is the desired (x,y,z) position and (x,y,z) translational velocity of the torso
        cage_max_dim = 11.75
        max_height = 1
        max_velo = 3
        subgoal_bounds = np.array([[-cage_max_dim,cage_max_dim],[-cage_max_dim,cage_max_dim],[0,max_height],[-max_velo, max_velo],[-max_velo, max_velo]])

        len_threshold = 0.5
        height_threshold = 0.2
        endgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold])

        # Set subgoal achievement thresholds
        velo_threshold = 0.5
        # subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, quat_threshold, quat_threshold, quat_threshold, quat_threshold, velo_threshold, velo_threshold, velo_threshold])
        subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, velo_threshold, velo_threshold])

        # Configs for agent
        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3
        agent_params["subgoal_penalty"] = -args.time_scale
        agent_params["random_action_perc"] = 0.3

        agent_params["atomic_noise"] = [0.2 for i in range(8)]
        agent_params["subgoal_noise"] = [0.2 for i in range(len(subgoal_thresholds))]

        agent_params["num_pre_training_episodes"] = 25

        agent_params["episodes_to_store"] = 500
        agent_params["num_exploration_episodes"] = 100

        #################### END CONFIGS #######################
        self.seed(seed)

        self.agent_params = agent_params

        self.name = model_name

        MODEL_PATH = ASSETS_PATH / self.name

        # Create Mujoco Simulation
        self.model = load_model_from_path(str(MODEL_PATH))
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel) # State will include (i) joint angles and (ii) joint velocities
        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds = self.sim.model.actuator_ctrlrange[:,1] # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.endgoal_dim = len(goal_space_test)
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        # Projection functions
        self.project_state_to_endgoal = project_state_to_endgoal
        self.project_state_to_subgoal = project_state_to_subgoal


        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]


        # End goal/subgoal thresholds
        self.endgoal_thresholds = endgoal_thresholds
        self.subgoal_thresholds = subgoal_thresholds

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_colors = ["Magenta","Green","Red","Blue","Cyan","Orange","Maroon","Gray","White","Black"]

        self.max_actions = max_actions

        self.steps_cnt = 0

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

    def get_next_goal(self, test):
        end_goal = np.zeros((len(self.goal_space_test)))

        if not test and self.goal_space_train is not None:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_train[i][0],self.goal_space_train[i][1])
        else:
            assert self.goal_space_test is not None, "Need goal space for testing. Set goal_space_test variable in \"design_env.py\" file"

            for i in range(len(self.goal_space_test)):
                end_goal[i] = np.random.uniform(self.goal_space_test[i][0],self.goal_space_test[i][1])

        self.display_endgoal(end_goal)

        return end_goal

    # Get state, which concatenates joint positions and velocities
    def get_state(self):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    # Reset simulation to state within initial state specified by user
    def reset(self, next_goal = None):

        self.steps_cnt = 0

        # Reset controls
        self.sim.data.ctrl[:] = 0

        while True:
            # Reset joint positions and velocities
            for i in range(len(self.sim.data.qpos)):
                self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

            for i in range(len(self.sim.data.qvel)):
                self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

            # Ensure initial ant position is more than min_dist away from goal
            min_dist = 8
            if np.linalg.norm(next_goal[:2] - self.sim.data.qpos[:2]) > min_dist:
                break

        self.sim.step()

        # Return state
        return self.get_state()

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):

        self.steps_cnt += 1

        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        return self.get_state(), 0.0, False, {}


    # Visualize end goal.  This function may need to be adjusted for new environments.
    def display_endgoal(self, end_goal):
        self.sim.data.mocap_pos[0][:2] = np.copy(end_goal[:2])

    # Visualize all subgoals
    def display_subgoals(self,subgoals):

        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1,min(len(subgoals),11)):
            self.sim.data.mocap_pos[i][:3] = np.copy(subgoals[subgoal_ind][:3])
            self.sim.model.site_rgba[i][3] = 1

            subgoal_ind += 1

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]
