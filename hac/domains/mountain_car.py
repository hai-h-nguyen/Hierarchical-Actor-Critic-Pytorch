# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np
import time

import gym
from gym import spaces
from gym.utils import seeding

try:
    from gym.envs.classic_control import rendering as visualize
except:
    pass

class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, args, seed, max_actions=1200, num_frames_skip=10, show=False):

        #################### START CONFIGS #######################
        # TODO: Correct?
        if args.n_layers in [1]:
            args.time_scale = 100
            max_actions = 100
        elif args.n_layers in [2]:
            args.time_scale = 20
            max_actions = 200
        elif args.n_layers in [3]:
            args.time_scale = 10
            max_actions = args.time_scale**args.n_layers

        self.action_dim = 1
        self.action_bounds = [1.0]
        self.action_offset = np.zeros((len(self.action_bounds)))

        self.goal_space_train = [[0.45, 0.48]]
        self.goal_space_test = [[0.45, 0.48]]
        self.endgoal_dim = len(self.goal_space_test)
        self.endgoal_thresholds = np.array([0.01])

        self.subgoal_bounds = np.array([[-1.2, 0.6],[-0.07, 0.07]])
        self.subgoal_dim = len(self.subgoal_bounds)

        # functions to project state to goal
        self.project_state_to_subgoal = lambda sim, state: state
        self.project_state_to_endgoal = lambda sim, state: [state[0]]

        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]

        self.subgoal_thresholds = np.array([0.01, 0.02])

        self.state_dim = 2
        self.name = "MountainCar"
        self.max_actions = max_actions

        self.initial_state_space = np.array([[-0.6, -0.4],[0.0, 0.0]])

        # Configs for agent
        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3
        agent_params["random_action_perc"] = 0.2

        agent_params["subgoal_penalty"] = -args.time_scale
        
        agent_params["atomic_noise"] = [0.1]
        agent_params["subgoal_noise"] = [0.02, 0.01]

        agent_params["episodes_to_store"] = 200
        agent_params["num_exploration_episodes"] = 50
        agent_params["num_pre_training_episodes"] = -1

        self.agent_params = agent_params
        self.sim = None
        #################### END CONFIGS #######################

        self.setup_view = False

        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None
        self.show = show

        self.n_layers = args.n_layers

        screen_width = 600
        screen_height = 400

        if self.show:
            self.viewer = visualize.Viewer(screen_width, screen_height)            

        world_width = self.max_position - self.min_position
        self.scale = screen_width/world_width

        self.steps_cnt = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.steps_cnt += 1

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power -0.0025 * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)

        reward = 0
        if done:
            reward = 100.0
        reward-= math.pow(action[0],2)*0.1

        self.state = np.array([position, velocity])

        if self.show:
            self.render()

        return self.state, reward, done, {}

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = visualize.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = visualize.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

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
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = visualize.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_next_goal(self, test):
        end_goal = np.zeros((len(self.goal_space_test)))

        if not test:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_train[i][0], self.goal_space_train[i][1])
        else:
            for i in range(len(self.goal_space_train)):
                end_goal[i] = np.random.uniform(self.goal_space_test[i][0], self.goal_space_test[i][1])

        # self.display_endgoal(end_goal)

        return end_goal  

    def reset(self):
        self.steps_cnt = 0
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def _setup_view(self):
        if self.viewer is not None and not self.setup_view:
            carwidth=40
            carheight=20

            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*self.scale, ys*self.scale))

            self.track = visualize.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = visualize.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(visualize.Transform(translation=(0, clearance)))
            self.cartrans = visualize.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = visualize.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(visualize.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = visualize.make_circle(carheight/2.5)
            backwheel.add_attr(visualize.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            
            if self.n_layers in [2, 3]:
                ################ Goal ################
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

            flagx = (self.goal_position-self.min_position)*self.scale
            flagy1 = self._height(self.goal_position)*self.scale
            flagy2 = flagy1 + 50
            flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = visualize.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

            self.setup_view = True        

    def display_endgoal(self, end_goal, mode="human"):

        self._setup_view()

        if self.show:
            return self.viewer.render(return_rgb_array = mode=='rgb_array')
        else:
            return

    def display_subgoals(self, subgoals, mode="human"):

        self._setup_view()

        if self.show:
            pos = self.state[0]
            self.cartrans.set_translation((pos-self.min_position)*self.scale, self._height(pos)*self.scale)
            self.cartrans.set_rotation(math.cos(3 * pos))
            
            if self.n_layers in [2, 3]:
                pos1 = subgoals[0][0]
                self.cartrans1.set_translation((pos1-self.min_position)*self.scale, self._height(pos1)*self.scale)
                self.cartrans1.set_rotation(math.cos(3 * pos1))

            if self.n_layers in [3]:
                pos2 = subgoals[1][0]
                self.cartrans2.set_translation((pos2-self.min_position)*self.scale, self._height(pos2)*self.scale)
                self.cartrans2.set_rotation(math.cos(3 * pos2))

            return self.viewer.render(return_rgb_array = mode=='rgb_array')        
        else:
            return

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
