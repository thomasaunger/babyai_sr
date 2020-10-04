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
        self.add_argument("--task-id-seed", action="store_true",
                            help="use the task id within a Slurm job array as the seed")
        self.add_argument("--procs", type=int, default=64,
                            help="number of processes (default: 64)")

        # Algorithm arguments
        self.add_argument("--frames-per-proc", type=int, default=160,
                            help="number of frames per process before update (default: 160)")
        self.add_argument("--discount", type=float, default=0.99,
                            help="discount factor (default: 0.99)")
        self.add_argument("--reward-scale", type=float, default=20.,
                            help="Reward scale multiplier")
        self.add_argument("--gae-lambda", type=float, default=0.99,
                            help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
        self.add_argument("--n", type=int, default=64,
                            help="communication interval (default: 64)")
        self.add_argument("--no-comm", action="store_true", default=False,
                            help="don't use communication")
        self.add_argument("--conventional", action="store_true", default=False,
                            help="reward sender using environment instead of receiver state-value estimate")
        self.add_argument("--archimedean", action="store_true", default=False,
                            help="use Archimedean receiver")
        self.add_argument("--informed-sender", action="store_true", default=False,
                            help="allows sender to see the instruction")

    def parse_args(self):
        """
        Parse the arguments and perform some basic validation
        """

        args = super().parse_args()

        # Set seed for all randomness sources
        if args.seed == 0:
            args.seed = np.random.randint(10000)
        if args.task_id_seed:
            args.seed = int(os.environ["SLURM_ARRAY_TASK_ID"])
            print("set seed to {}".format(args.seed))

        return args
