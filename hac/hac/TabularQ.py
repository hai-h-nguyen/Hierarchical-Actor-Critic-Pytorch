import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ..utils import rand_argmax
import pickle as cpickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
class Critic(nn.Module):
    def __init__(self, 
        env, 
        layer_number, 
        args):

        super(Critic, self).__init__()

        # TODO: Make this a parameter
        self.state_size = 121

        self.layer_number = layer_number

        if layer_number == 1:
            self.action_size = self.state_size
            self.q_limit = -self.state_size
            # Q table for a fixed goal
            # Size: state x action
            self.Q_table = np.ones((self.state_size, self.action_size)) * self.q_limit
        else:
            self.action_size = env.action_size 

            self.q_limit = -args.time_scale
            # Q table for a fixed goal
            # Size: state x action
            self.Q_table = np.ones((self.state_size, self.state_size, self.action_size)) * self.q_limit
          

    def get_action(self, state, goal, greedy):
        if greedy:
            if self.layer_number == 1:
                action = rand_argmax(self.Q_table[state], goal)
            else:
                action = rand_argmax(self.Q_table[goal][state])
        else:
            if self.layer_number == 1:
                action = rand_argmax(self.Q_table[state], goal)
            else:
                action = np.random.randint(0, self.Q_table[goal].shape[1])

        return action

class TabularQ:
    def __init__(self, env, layer_number, args, lr=1.0, gamma=1.0):

        self.critic = Critic(env, layer_number, args)
        self.gamma = gamma
        self.lr = lr
        self.layer_number = layer_number
    
    def select_action(self, state, goal, greedy):
        
        return self.critic.get_action(state, goal=goal, greedy=greedy)
    
    def update(self, buffer):
        
        # Sample a batch of transitions from replay buffer:
        state, action, reward, next_state, goal, done = buffer.get_batch()
        
        # Update Q-table
        if self.layer_number == 0:
            for i in range(len(state)):
                target = reward[i] + self.gamma * (1 - done[i]) * np.max(self.critic.Q_table[goal[i]][next_state[i]])
                self.critic.Q_table[goal[i]][state[i]][action[i]] += self.lr * (target - self.critic.Q_table[goal[i]][state[i]][action[i]])
        else:
            for i in range(len(state)):
                target = reward[i] + self.gamma * (1 - done[i]) * np.max(self.critic.Q_table[next_state[i]])
                self.critic.Q_table[state[i], action[i]] += self.lr * (target - self.critic.Q_table[state[i], action[i]])

    def save(self, directory, name):
        cpickle.dump(self.critic.Q_table, open('%s/%s_critic.p' % (directory, name), 'wb'))
        
    def load(self, directory, name):
        self.critic.Q_table = cpickle.load(open('%s/%s_critic.p' % (directory, name), 'rb'))  
        
        
        
        
      
        
        
