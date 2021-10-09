# -*- coding: utf-8 -*-

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import socket
if socket.gethostname() in ['hainh-sys']:
    from gym.envs.classic_control import rendering as visualize

class CarEnv(gym.Env):
    def __init__(self, args=None, seed=0, max_actions=1200, show=False):
        self.max_position = 1.1
        self.min_position = -self.max_position
        self.max_speed = 0.07

        #################### START CONFIGS #######################
        if args is not None:
            if args.n_layers in [1]:
                args.time_scale = 100
                max_actions = 100
            elif args.n_layers in [2]:
                args.time_scale = 20
                max_actions = 160
            elif args.n_layers in [3]:
                args.time_scale = 10
                max_actions = 200

        self.args = args

        self.goal_space_train = [[self.min_position, self.max_position],[-self.max_speed, self.max_speed]]
        self.goal_space_test = [[self.min_position, self.max_position],[0.0, 0.0]]

        self.endgoal_dim = 2

        self.action_dim = 1
        self.action_bounds = [1.0]
        self.action_offset = np.zeros((len(self.action_bounds)))

        self.subgoal_bounds = np.array([[self.min_position, self.max_position],[-self.max_speed, self.max_speed]])
        self.subgoal_dim = len(self.subgoal_bounds)

        # functions to project state to goal
        self.project_state_to_endgoal = lambda sim, state: state[:-1]
        self.project_state_to_subgoal = lambda sim, state: state[:-1]
        

        self.prepare_high_obs_fn = lambda state: state
        self.reduced_obs_dim = args.time_scale*3 # args.time_scale x self.state_dim

        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        self.subgoal_thresholds = np.array([0.01, 0.01])
        self.endgoal_thresholds = np.array([0.01, 0.01])

        self.state_dim = 3
        self.name = "Car-Flag-MDP"
        self.max_actions = max_actions

        # Configs for agent
        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3

        agent_params["random_action_perc"] = 0.2
        agent_params["num_pre_training_episodes"] = 10

        if args is not None:
            agent_params["subgoal_penalty"] = -args.time_scale
        
        agent_params["atomic_noise"] = [0.1]
        agent_params["subgoal_noise"] = [0.03, 0.03]

        agent_params["episodes_to_store"] = 200
        agent_params["num_exploration_episodes"] = 50

        self.agent_params = agent_params
        self.sim = None
        #################### END CONFIGS #######################
        self.setup_view = False

        self.min_action = -1.0
        self.max_action = 1.0

        self.heaven_position = 1.0
        self.hell_position = -1.0
        self.priest_position = 0.5
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None
        self.show = show

        self.screen_width = 600
        self.screen_height = 400

        # When the cart is within this vicinity, it observes the direction given
        # by the priest
        self.priest_delta = 0.2

        self.low_state = np.array(
            [self.min_position, -self.max_speed, -1.0], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed, 1.0], dtype=np.float32
        )

        world_width = self.max_position - self.min_position
        self.scale = self.screen_width/world_width

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        if args is not None:
            self.n_layers = args.n_layers

        self.done = False
        self.solved = False
        self.steps_cnt = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if np.isscalar(action):
            action = [action]

        self.steps_cnt += 1

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force * self.power
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        max_position = max(self.heaven_position, self.hell_position)
        min_position = min(self.heaven_position, self.hell_position)

        done = bool(
            position >= max_position or position <= min_position
        )

        self.done = done

        reward = 0.0
        if (self.heaven_position > self.hell_position):
            if (position >= self.heaven_position):
                reward = 1.0

        if (self.heaven_position < self.hell_position):
            if (position <= self.heaven_position):
                reward = 1.0

        direction = 0.0
        if position >= self.priest_position - self.priest_delta and position <= self.priest_position + self.priest_delta:
            if (self.heaven_position > self.hell_position):
                # Heaven on the right
                direction = 1.0
            else:
                # Heaven on the left
                direction = -1.0

        self.state = np.array([position, velocity, self.heaven_position])
        self.solved = (reward > 0.0)

        if self.show:
            self.render()

        env_reward = -1
        if self.solved:
            env_reward = 0

        return self.state, env_reward, done, {}

    def render(self, mode='human'):
        self._setup_view()

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * self.scale, self._height(pos) * self.scale
        )

        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def reset(self):

        self.solved = False
        self.done = False
        self.steps_cnt = 0

        # Randomize the heaven/hell location
        if (self.np_random.randint(2) == 0):
            self.heaven_position = 1.0
        else:
            self.heaven_position = -1.0

        self.hell_position = -self.heaven_position

        if self.viewer is not None:
            self._draw_flags()
            self._draw_boundary()

        self.state = np.array([self.np_random.uniform(low=-0.2, high=0.2), 0, self.heaven_position])
        return np.array(self.state)

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

    def _height(self, xs):
        return .55 * np.ones_like(xs)

    def _draw_boundary(self):
        flagx = (self.priest_position-self.priest_delta-self.min_position)*self.scale
        flagy1 = self._height(self.priest_position)*self.scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)            

        flagx = (self.priest_position+self.priest_delta-self.min_position)*self.scale
        flagy1 = self._height(self.priest_position)*self.scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)         

    def _draw_flags(self):
        scale = self.scale
        # Flag Heaven
        flagx = (abs(self.heaven_position)-self.min_position)*scale
        flagy1 = self._height(self.heaven_position)*scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )

        # RED for hell
        if self.heaven_position > self.hell_position:
            flag.set_color(0.0, 1.0, 0)
        else:
            flag.set_color(1.0, 0.0, 0)

        self.viewer.add_geom(flag)

        # Flag Hell
        flagx = (-abs(self.heaven_position)-self.min_position)*scale
        flagy1 = self._height(self.hell_position)*scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )

        # GREEN for heaven
        if self.heaven_position > self.hell_position:
            flag.set_color(1.0, 0.0, 0)
        else:
            flag.set_color(0.0, 1.0, 0)

        self.viewer.add_geom(flag)

        # BLUE for priest
        flagx = (self.priest_position-self.min_position)*scale
        flagy1 = self._height(self.priest_position)*scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )
        flag.set_color(0.0, 0.0, 1.0)
        self.viewer.add_geom(flag)

    def _setup_view(self):
        if  not self.setup_view:
            self.viewer = visualize.Viewer(self.screen_width, self.screen_height)
            scale = self.scale
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = visualize.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10
            carwidth = 40
            carheight = 20

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = visualize.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(visualize.Transform(translation=(0, clearance)))
            self.cartrans = visualize.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = visualize.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                visualize.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = visualize.make_circle(carheight / 2.5)
            backwheel.add_attr(
                visualize.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)

            self._draw_flags()
            self._draw_boundary()

            if self.args is not None:

                ################ End Goal ################
                car0 = visualize.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                car0.set_color(0.0, 0.0, 1.0)
                car0.add_attr(visualize.Transform(translation=(0, clearance)))
                self.cartrans0 = visualize.Transform()
                car0.add_attr(self.cartrans0)
                self.viewer.add_geom(car0)
                ######################################

                if self.n_layers in [2, 3]:
                
                    ################ Goal 1 ################
                    car1 = visualize.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                    car1.set_color(1, 0.0, 0.0)
                    car1.add_attr(visualize.Transform(translation=(0, clearance)))
                    self.cartrans1 = visualize.Transform()
                    car1.add_attr(self.cartrans1)
                    self.viewer.add_geom(car1)
                    ######################################

                if self.n_layers in [3]:

                    ############### Goal 2 ###############
                    car2 = visualize.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                    car2.set_color(0.0, 1, 0.0)
                    car2.add_attr(visualize.Transform(translation=(0, clearance)))
                    self.cartrans2 = visualize.Transform()
                    car2.add_attr(self.cartrans2)
                    self.viewer.add_geom(car2)
                    ######################################

            self.setup_view = True        

    def display_endgoal(self, endgoal):

        if self.show:
            self._setup_view()
            pos = self.state[0]
            self.cartrans.set_translation((pos-self.min_position)*self.scale, self._height(pos)*self.scale)
            
            pos1 = endgoal[0]
            self.cartrans0.set_translation((pos1-self.min_position)*self.scale, self._height(pos1)*self.scale)

            return self.viewer.render()        
        else:
            return

    def display_subgoals(self, subgoals, mode="human"):

        self._setup_view()

        if self.show:
            pos = self.state[0]
            self.cartrans.set_translation((pos-self.min_position)*self.scale, self._height(pos)*self.scale)
            
            if self.n_layers in [2, 3]:
                pos1 = subgoals[0][0]
                self.cartrans1.set_translation((pos1-self.min_position)*self.scale, self._height(pos1)*self.scale)

            if self.n_layers in [3]:
                pos2 = subgoals[1][0]
                self.cartrans2.set_translation((pos2-self.min_position)*self.scale, self._height(pos2)*self.scale)

            return self.viewer.render(return_rgb_array=mode=='rgb_array')        
        else:
            return

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None