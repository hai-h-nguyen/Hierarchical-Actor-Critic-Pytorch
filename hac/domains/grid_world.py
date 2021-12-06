from tkinter import *
import time
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class Grid_World(gym.Env):

    def __init__(self, args=None, seed=0, max_actions=100, show=False):

        if args.n_layers in [2]:
            args.time_scale = 5
            max_actions = 50

        # functions to project state to goal
        project_state_to_endgoal = lambda state: state
        project_state_to_subgoal = lambda state: state

        self.name = 'Grid-World'

        self.visualize = show

        world_len = 11
        state_mat = np.ones((world_len,world_len))

        state_mat[:,5] = 0
        state_mat[5,:] = 0
        state_mat[5,2:4] = 1
        state_mat[7:9,5] = 1
        state_mat[5,7:9] = 1

        goal_row = 2
        goal_col = 7
        self.goal = goal_row * state_mat.shape[1] + goal_col

        # Create top-level window
        if self.visualize:
            self.root = Tk()
            self.root.title("Grid World")

            # Create canvas to hold grid world
            self.canvas = Canvas(self.root,width = "500",height = "500")
            self.canvas.grid(column=0, row=0, sticky=(N, W, E, S))

        # Create grid world
        self.state_mat = state_mat 

        # Determine pixel length of each block (assume state_mat is square)       
        num_col = state_mat.shape[1]
        pixel_width = 480
        while pixel_width % num_col != 0:
            pixel_width -= 1
        # print("Block Pixel Length: ", pixel_width)

        num_row = state_mat.shape[0]

        block_length = pixel_width / num_col

        self.action_space = spaces.Discrete(4)
            
        # Create rectangles    
        for i in range(num_row):
            for j in range(num_col):
                x_1 = 10 + block_length * j
                y_1 = 10 + block_length * i
                x_2 = x_1 + block_length
                y_2 = y_1 + block_length

                if self.state_mat[i][j] == 1:
                    color = "white"
                else:
                    color = "black"
                
                if self.visualize:
                    self.canvas.create_rectangle(x_1,y_1,x_2,y_2,fill=color)

        self.steps_cnt = 0

        # Configs for agent
        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3
        agent_params["subgoal_penalty"] = -args.time_scale
        agent_params["random_action_perc"] = 0.3


        agent_params["num_pre_training_episodes"] = -1
        agent_params["episodes_to_store"] = 500
        agent_params["num_exploration_episodes"] = 100

        self.agent_params = agent_params

        self.action_size = 4
        self.state_dim = 1
        self.endgoal_dim = 1
        self.subgoal_dim = 1

        self.project_state_to_endgoal = project_state_to_endgoal
        self.project_state_to_subgoal = project_state_to_subgoal

        self.max_actions = max_actions
        self.old_subgoal_1 = -1
        self.endgoal_1 = -1

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def get_next_goal(self, test):
        """
        Randomize the position of the goal (the agent starts at a fixed location)
        """
        return self.goal


    def reset(self):

        self.state = 0
        self.steps_cnt = 0

        # Reset blocks to original colors
        for i in range(self.state_mat.shape[0]):
            for j in range(self.state_mat.shape[1]):

                if self.state_mat[i][j] == 1:
                    color = "white"
                else:
                    color = "black"

                id_num = i * len(self.state_mat[0]) + j + 1
                    
                if self.visualize:
                    self.canvas.itemconfig(id_num,fill=color)
        
        if self.visualize:
            # Change color of agent's current state and goal state
            self.canvas.itemconfig(self.state + 1, fill="blue")
            self.canvas.itemconfig(self.goal + 1, fill="yellow")

            self.root.update()
            time.sleep(0.1)

        return self.get_state()

    def display_subgoals(self, subgoals):

        state = self.state

        if subgoals[0] != self.old_subgoal_1:
            if self.old_subgoal_1 != -1:
                # If agent currently at old subgoal
                if self.old_subgoal_1 == state:
                    self.canvas.itemconfig(self.old_subgoal_1 + 1,fill="blue")
                # If at the end-goal, then paint the goal yellow
                elif self.old_subgoal_1 == self.goal:
                    self.canvas.itemconfig(self.old_subgoal_1 + 1,fill="yellow")
                # Remove the color of the last subgoal
                else:
                    self.canvas.itemconfig(self.old_subgoal_1 + 1,fill="white")  
                                       
            # print(self.old_subgoal_1)
            self.canvas.itemconfig(subgoals[0] + 1,fill="magenta") 
            self.root.update()

        self.old_subgoal_1 = subgoals[0]
        time.sleep(0.1)


    def step(self, action):

        old_state = np.copy(self.state)
        new_state = self.get_next_state(action)

        self.steps_cnt += 1

        reward = -1.0

        if self.visualize:
            # If state has changed, update blocks colors
            if new_state != old_state:
                self.canvas.itemconfig(old_state + 1,fill="white") 
            
            if new_state != self.goal:            
                self.canvas.itemconfig(new_state + 1,fill="blue")
            else:
                self.canvas.itemconfig(new_state + 1,fill="orange")
                reward = 0.0

            self.root.update()

            time.sleep(0.1)

        self.state = new_state

        return self.get_state(), reward, False, {}

    def get_state(self):
        return self.state

    def get_next_state(self, action):

        state = self.state
        state_mat = self.state_mat

        num_col = state_mat.shape[1]
        num_row = state_mat.shape[0]

        state_row = int(state/num_col)    
        state_col = state % num_col
        
        # print("State row: ", state_row)
        # print("State col: ", state_col)

        # If action is "left"
        if action == 0:
            if state_col != 0 and state_mat[state_row][state_col - 1] == 1:
                # print("Moving Left")
                state -= 1
        # If action is "up"
        elif action == 1:
            if state_row != 0 and state_mat[state_row - 1][state_col] == 1:
                # print("Moving Up")
                state -= num_col
        # If action is "right"
        elif action == 2:
            if state_col != (num_col - 1) and state_mat[state_row][state_col + 1] == 1:
                # print("Moving Right")
                state += 1
        # If action is "down"
        else:
            if state_row != (num_row - 1) and state_mat[state_row + 1][state_col] == 1:
                # print("Moving Down")
                state += num_col

        return state
