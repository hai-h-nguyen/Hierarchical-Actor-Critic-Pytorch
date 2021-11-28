from tkinter import *
import time
import numpy as np
import gym
from gym import spaces

class Grid_World(gym.Env):

    def __init__(self):

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
                    
                self.canvas.create_rectangle(x_1,y_1,x_2,y_2,fill=color)
        
    def reset(self):

        self.state = 0

        # Reset blocks to original colors
        for i in range(self.state_mat.shape[0]):
            for j in range(self.state_mat.shape[1]):

                if self.state_mat[i][j] == 1:
                    color = "white"
                else:
                    color = "black"

                id_num = i * len(self.state_mat[0]) + j + 1
                    
                self.canvas.itemconfig(id_num,fill=color)

        # Change color of agent's current state and goal state
        self.canvas.itemconfig(self.state + 1, fill="blue")
        self.canvas.itemconfig(self.goal + 1, fill="yellow")

        self.root.update()
        # time.sleep(0.1)

        return self.get_state()

        


    def display_subgoals(self, subgoal_1, old_subgoal_1, state, goal):
   
        if subgoal_1 != old_subgoal_1:
            if old_subgoal_1 != -1:
                # If agent currently at old subgoal
                if old_subgoal_1 == state:
                    self.canvas.itemconfig(old_subgoal_1 + 1,fill="blue")
                elif old_subgoal_1 == goal:
                    self.canvas.itemconfig(old_subgoal_1 + 1,fill="yellow")
                else:
                    self.canvas.itemconfig(old_subgoal_1 + 1,fill="white")  
                                       
            
            self.canvas.itemconfig(subgoal_1 + 1,fill="magenta") 
            self.root.update()
        time.sleep(0.1)

            # 

    def step(self, action):

        old_state = np.copy(self.state)
        new_state = self.get_next_state(action)

        reward = -1.0

        # If state has changed, update blocks colors
        if new_state != old_state:
            self.canvas.itemconfig(old_state + 1,fill="white") 
        
        if new_state != self.goal:            
            self.canvas.itemconfig(new_state + 1,fill="blue")
        else:
            self.canvas.itemconfig(new_state + 1,fill="orange")
            reward = 0.0

        self.root.update()

        # time.sleep(0.2)

        self.state = new_state

        # TODO: return correctly
        return self.get_state(), reward, False, {}

    def get_state(self):
        return self.state / 121.0

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
