import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
from gym import spaces
from pathlib import Path
from gym.utils import seeding


ASSETS_PATH = Path(__file__).resolve().parent / 'assets'

def bound_angle_ur5(angle):

    bounded_angle = np.absolute(angle) % (2*np.pi)
    if angle < 0:
        bounded_angle = -bounded_angle

    return bounded_angle


class UR5Env(gym.Env):

    def __init__(self, args, seed, max_actions=1200, num_frames_skip=10, show=False):

        #################### START CONFIGS #######################
        if args.n_layers in [1]:
            args.time_scale = 600
            max_actions = 600
        elif args.n_layers in [2]:
            args.time_scale = 25
            max_actions = 600
        elif args.n_layers in [3]:
            args.time_scale = 10
            max_actions = args.time_scale**(args.n_layers-1)*6

        timesteps_per_action = 15
        num_frames_skip = timesteps_per_action

        model_name = "ur5.xml"
        initial_joint_pos = np.array([  5.96625837e-03,   3.22757851e-03,  -1.27944547e-01])
        initial_joint_pos = np.reshape(initial_joint_pos,(len(initial_joint_pos),1))
        initial_joint_ranges = np.concatenate((initial_joint_pos,initial_joint_pos),1)
        initial_joint_ranges[0] = np.array([-np.pi/8,np.pi/8])

        initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges),2))),0)
        goal_space_train = [[-np.pi,np.pi],[-np.pi/4,0],[-np.pi/4,np.pi/4]]
        goal_space_test = [[-np.pi,np.pi],[-np.pi/4,0],[-np.pi/4,np.pi/4]]


        project_state_to_endgoal = lambda sim, state: np.array([bound_angle_ur5(sim.data.qpos[i]) for i in range(len(sim.data.qpos))])
        angle_threshold = np.deg2rad(10)
        endgoal_thresholds = np.array([angle_threshold, angle_threshold, angle_threshold])

        subgoal_bounds = np.array([[-2*np.pi,2*np.pi],[-2*np.pi,2*np.pi],[-2*np.pi,2*np.pi],[-4,4],[-4,4],[-4,4]])
        project_state_to_subgoal = lambda sim, state: np.concatenate((np.array([bound_angle_ur5(sim.data.qpos[i]) for i in range(len(sim.data.qpos))]),
                                    np.array([4 if sim.data.qvel[i] > 4 else -4 if sim.data.qvel[i] < -4 else sim.data.qvel[i] for i in range(len(sim.data.qvel))])))

        velo_threshold = 2
        subgoal_thresholds = np.concatenate((np.array([angle_threshold for i in range(3)]), np.array([velo_threshold for i in range(3)])))

        # Configs for agent
        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3
        agent_params["random_action_perc"] = 0.2
        agent_params["subgoal_penalty"] = -args.time_scale
        agent_params["atomic_noise"] = [0.1 for i in range(3)]
        agent_params["subgoal_noise"] = [0.03 for i in range(6)]
        agent_params["episodes_to_store"] = 500
        agent_params["num_exploration_episodes"] = 50
        agent_params["num_pre_training_episodes"] = -1

        #################### END CONFIGS #######################

        self.seed(seed)

        self.agent_params = agent_params

        self.name = model_name

        MODEL_PATH = ASSETS_PATH / self.name

        # Create Mujoco Simulation
        self.model = load_model_from_path(str(MODEL_PATH))
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        if model_name == "pendulum.xml":
            self.state_dim = 2*len(self.sim.data.qpos) + len(self.sim.data.qvel)
        else:
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

        # Implement visualization if necessary
        self.visualize = show  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

        self.steps_cnt = 0
        
    # Get state, which concatenates joint positions and velocities
    def get_state(self):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    # Reset simulation to state within initial state specified by user
    def reset(self):

        self.steps_cnt = 0

        # Reset joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = self.np_random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = self.np_random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

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
    def display_end_goal(self, end_goal):

        joint_pos = self._angles2jointpos(end_goal[:3])

        for i in range(3):
            self.sim.data.mocap_pos[i] = joint_pos[i]

    def get_next_goal(self, test):

        goal_possible = False

        while not goal_possible:
            end_goal = np.zeros(shape=(self.endgoal_dim,))

            end_goal[0] = np.random.uniform(self.goal_space_test[0][0],self.goal_space_test[0][1])
            end_goal[1] = np.random.uniform(self.goal_space_test[1][0],self.goal_space_test[1][1])
            end_goal[2] = np.random.uniform(self.goal_space_test[2][0],self.goal_space_test[2][1])

            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            # shoulder_pos_1 = np.array([0,0,0,1])
            upper_arm_pos_2 = np.array([0,0.13585,0,1])
            forearm_pos_3 = np.array([0.425,0,0,1])
            wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])

            # Transformation matrix from shoulder to base reference frame
            T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

            # Transformation matrix from upper arm to shoulder reference frame
            T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0],[np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

            # Transformation matrix from forearm to upper arm reference frame
            T_3_2 = np.array([[np.cos(theta_2),0,np.sin(theta_2),0],[0,1,0,0.13585],[-np.sin(theta_2),0,np.cos(theta_2),0],[0,0,0,1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            T_4_3 = np.array([[np.cos(theta_3),0,np.sin(theta_3),0.425],[0,1,0,0],[-np.sin(theta_3),0,np.cos(theta_3),0],[0,0,0,1]])

            forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
            wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

            # Make sure wrist 1 pos is above ground so can actually be reached
            if np.absolute(end_goal[0]) > np.pi/4 and forearm_pos[2] > 0.05 and wrist_1_pos[2] > 0.15:
                goal_possible = True

        self.display_end_goal(end_goal)

        return end_goal

    # Visualize all subgoals
    def display_subgoals(self,subgoals):

        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11

        for i in range(1,min(len(subgoals),11)):
            angles = subgoals[subgoal_ind][:3]
            joint_pos = self._angles2jointpos(angles)

            # Designate site position for upper arm, forearm and wrist
            for j in range(3):     
                self.sim.data.mocap_pos[3 + 3*(i-1) + j] = np.copy(joint_pos[j])
                self.sim.model.site_rgba[3 + 3*(i-1) + j][3] = 1

            subgoal_ind += 1

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def _angles2jointpos(self, angles):
        theta_1 = angles[0]
        theta_2 = angles[1]
        theta_3 = angles[2]

        upper_arm_pos_2 = np.array([0,0.13585,0,1])
        forearm_pos_3 = np.array([0.425,0,0,1])
        wrist_1_pos_4 = np.array([0.39225,-0.1197,0,1])

        # Transformation matrix from shoulder to base reference frame
        T_1_0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.089159],[0,0,0,1]])

        # Transformation matrix from upper arm to shoulder reference frame
        T_2_1 = np.array([[np.cos(theta_1), -np.sin(theta_1), 0, 0], [np.sin(theta_1), np.cos(theta_1), 0, 0],[0,0,1,0],[0,0,0,1]])

        # Transformation matrix from forearm to upper arm reference frame
        T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                        [0, 1 ,0 , 0.13585], 
                        [-np.sin(theta_2),0,np.cos(theta_2),0],
                        [0,0,0,1]])

        # Transformation matrix from wrist 1 to forearm reference frame
        T_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3),0.425], 
                [0,1,0,0], 
                [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                [0,0,0,1]])

        # Determine joint position relative to original reference frame
        upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
        forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
        wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

        joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

        return joint_pos