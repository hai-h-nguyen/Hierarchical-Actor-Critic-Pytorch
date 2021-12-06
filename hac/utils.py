import numpy as np
import random
import torch
import os

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


def rand_argmax(arr, goal = None):

    curr_max = arr[0]
    max_ind = [0]

    for i in range(1,len(arr)):
        if arr[i] >= curr_max:
            if arr[i] > curr_max:
                max_ind = [i]
                curr_max = arr[i]
            else:
                max_ind.append(i)

    # Return index from max_ind array
    if goal is None:
        return_ind = np.random.randint(0,len(max_ind))
    else:
        # Check if goal state is in list
        for i in range(len(max_ind)):
            if max_ind[i] == goal:
                return max_ind[i]
        return_ind = np.random.randint(0,len(max_ind))

        
    return max_ind[return_ind]
