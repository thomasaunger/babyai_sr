"""
Common arguments for BabyAI training scripts
"""

import os
import argparse
import numpy as np


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        # Base arguments
        self.add_argument("--env", default=None,
                            help="name of the environment to train on (REQUIRED)")
        self.add_argument("--seed", type=int, default=1,
                            help="random seed; if 0, a random random seed will be used  (default: 1)")
        self.add_argument("--task-id-seed", action='store_true',
                            help="use the task id within a Slurm job array as the seed")
        self.add_argument("--procs", type=int, default=64,
                            help="number of processes (default: 64)")
        self.add_argument("--tb", action="store_true", default=False,
                            help="log into Tensorboard")

        # Training arguments
        self.add_argument("--log-interval", type=int, default=1,
                            help="number of updates between two logs (default: 1)")
        self.add_argument("--frames", type=int, default=int(9e10),
                            help="number of frames of training (default: 9e10)")
        self.add_argument("--frames-per-proc", type=int, default=160,#40,
                            help="number of frames per process before update (default: 160)")
        self.add_argument("--lr", type=float, default=1e-4,
                            help="learning rate (default: 1e-4)")
        self.add_argument("--beta1", type=float, default=0.9,
                            help="beta1 for Adam (default: 0.9)")
        self.add_argument("--beta2", type=float, default=0.999,
                            help="beta2 for Adam (default: 0.999)")
        self.add_argument("--recurrence", type=int, default=160,
                            help="number of timesteps gradient is backpropagated (default: 160)")
        self.add_argument("--optim-eps", type=float, default=1e-5,
                            help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
        self.add_argument("--batch-size", type=int, default=5120,
                                help="batch size for PPO (default: 5120)")
        self.add_argument("--entropy-coef", type=float, default=0.01,
                            help="entropy term coefficient (default: 0.01)")

        # Model parameters
        self.add_argument("--image-dim", type=int, default=512,
                            help="dimensionality of the image embedding")
        self.add_argument("--memory-dim", type=int, default=1024,
                            help="dimensionality of the memory LSTM")
        self.add_argument("--instr-dim", type=int, default=512,
                            help="dimensionality of the memory LSTM")
        self.add_argument("--enc-dim", type=int, default=512,
                            help="dimensionality of the encoder LSTM")
        self.add_argument("--dec-dim", type=int, default=512,
                            help="dimensionality of the decoder LSTM")

    def parse_args(self):
        """
        Parse the arguments and perform some basic validation
        """

        args = super().parse_args()

        # Set seed for all randomness sources
        if args.seed == 0:
            args.seed = np.random.randint(10000)
        if args.task_id_seed:
            args.seed = int(os.environ['SLURM_ARRAY_TASK_ID'])
            print('set seed to {}'.format(args.seed))

        return args
