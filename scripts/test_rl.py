#!/usr/bin/env python3

"""
Script to test a sender and receiver through reinforcement learning.
"""

import gym
import time
import datetime
import torch
import numpy as np

import babyai.utils as utils

from babyai_sr.arguments     import ArgumentParser
from babyai_sr.rl.algos.test import TestAlgo
from babyai_sr.rl.utils.penv import ParallelEnv

import babyai_sr.levels.sr_levels

# Parse arguments.
parser = ArgumentParser()

parser.add_argument("--sender", default=None,
                    help="name of the sender (REQUIRED)")
parser.add_argument("--receiver", default=None,
                    help="name of the receiver (REQUIRED)")
parser.add_argument("--sample", action="store_true", default=False,
                    help="sample messages instead of using argmax")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes to test on (default: 1000)")

args = parser.parse_args()

utils.seed(args.seed)

envs = []
for i in range(args.procs):
    env = gym.make(args.env)
    env.seed(100 * args.seed + i)
    envs.append(env)

penv = ParallelEnv(envs, args.n, args.conventional)

# Define obss preprocessor.
obss_preprocessor = utils.ObssPreprocessor(args.receiver, envs[0].observation_space)

# Define actor--critic models.
sender   = utils.load_model(args.sender)
receiver = utils.load_model(args.receiver)

if torch.cuda.is_available():
    sender.cuda()
    receiver.cuda()

# Define actor--critic algorithm.
reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
test_algo = TestAlgo(penv, [sender, receiver], args.frames_per_proc, args.discount, args.gae_lambda, obss_preprocessor, reshape_reward, not args.no_comm, args.conventional, args.archimedean, not args.sample, args.ignorant_sender)

# Test models.
sender.eval()
receiver.eval()
total_start_time = time.time()
update_start_time = time.time()
exp_data, logs = test_algo.collect_episodes(args.episodes)
update_end_time = time.time()

# Print log.
total_ellapsed_time = int(time.time() - total_start_time)
fps = logs["num_frames"] / (update_end_time - update_start_time)
duration = datetime.timedelta(seconds=total_ellapsed_time)
return_per_episode = utils.synthesize(logs["return_per_episode"])
success_per_episode = utils.synthesize(
    [1 if r > 0 else 0 for r in logs["return_per_episode"]])
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

data = [logs["episodes_done"], logs["num_frames"],
        fps, total_ellapsed_time,
        *return_per_episode.values(),
        success_per_episode["mean"],
        *num_frames_per_episode.values()]

format_str = ("E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
              "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | ")

np.save("data.npy", exp_data.numpy())

print(format_str.format(*data))
