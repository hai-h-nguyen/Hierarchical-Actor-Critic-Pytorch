import argparse

"""
Below are training options user can specify in command line.

Options Include:

1. Retrain boolean
- If included, actor and critic neural network parameters are reset

2. Testing boolean
- If included, agent only uses greedy policy without noise.  No changes are made to policy and neural networks. 
- If not included, periods of training are by default interleaved with periods of testing to evaluate progress.

3. Show boolean
- If included, training will be visualized

4. Train Only boolean
- If included, agent will be solely in training mode and will not interleave periods of training and testing

5. Verbosity boolean
- If included, summary of each transition will be printed
"""

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Include to reset policy'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Include to fix current policy'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Include to visualize training'
    )

    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Include to use training mode only'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print summary of each transition'
    )

    parser.add_argument(
        '--log-level',
        type=int,
        default=40,
        help='Log level: 10: Debug, 20: Info, 30: Warn, 40: Err, 50: Critical'
    )

    parser.add_argument(
        '--noisy-obs',
        action='store_true',
        help='Noisy observation'
    )

    parser.add_argument(
        '--random-act',
        action='store_true',
        help='Random action with some probability'
    )  

    parser.add_argument(
        '--n_layers',
        type=int,
        default=2,
        help='Number of layers: default 2'
    )

    parser.add_argument(
        '--num-batch-train',
        type=int,
        default=1000000,
        help='Number of batches to train: default 1000'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Number of training steps: default 200000'
    )    

    parser.add_argument(
        '--test-freq',
        type=int,
        default=2,
        help='Test frequency: default 2'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed: default 0'
    )

    parser.add_argument(
        '--env',
        type=str,
        default='pendulum-v0',
        help='Name of the environment'
    )    

    parser.add_argument(
        '--group',
        type=str,
        default=None,
        help='wandb group name'
    ) 

    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='wandb run name'
    ) 

    FLAGS, unparsed = parser.parse_known_args()


    return FLAGS
