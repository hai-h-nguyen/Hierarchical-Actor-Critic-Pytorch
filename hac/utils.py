import numpy as np
import random
import torch
import os
import gym

# Below function prints out options and environment specified by user
def print_summary(args,env):

    print("\n- - - - - - - - - - -")
    print("Task Summary: ","\n")
    print("Environment: ", env.name)
    print("Number of Layers: ", args.n_layers)
    print("Time Limit per Layer: ", args.time_scale)
    print("Max Episode Time Steps: ", env.max_actions)
    print("Retrain: ", args.retrain)
    print("Test: ", args.test)
    print("Visualize: ", args.show)
    print("- - - - - - - - - - -", "\n\n")

def init_weights(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        # Inner linear layer
        if m.out_features > 1:
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
        
        # Output linear layer
        else:
            torch.nn.init.uniform_(m.weight, a=-3e-3, b=3e-3)
            torch.nn.init.uniform_(m.bias, a=-3e-3, b=3e-3)

class NoisyObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_p=0.9, noise_delta=0.1, untouched_lst=[]):
        """
        Replace o_t with a uniform sample in [o_t*(1-self.delta), o_t*(1+self.delta)] with a probability self.p
        """
        super(NoisyObsWrapper, self).__init__(env)
        self.p = noise_p
        self.delta = noise_delta
        self.untouched_lst = untouched_lst

    def observation(self, observation):
        if np.random.binomial(1, self.p, 1).item():
            observation_perturbed = np.random.uniform(low=observation*(1-self.delta), high=observation*(1+self.delta), size=observation.shape)

            for i in self.untouched_lst:
                observation_perturbed[i] = observation[i]

            return observation_perturbed

        return observation

class RandomActWrapper(gym.ActionWrapper):
    def __init__(self, env, noise_p=0.1):
        """
        Replace a_t with a uniform sample in [-1.0, 1.0] with a probability p
        """
        super(RandomActWrapper, self).__init__(env)
        self.p = noise_p

    def action(self, action):
        if random.random() < self.p:
            return self.env.action_space.sample()
        return action