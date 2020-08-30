#!/usr/bin/env python3

"""
Script to train a sender and receiver through reinforcement learning.
"""

import os
import logging
import csv
import json
import gym
import time
import datetime
import torch
import subprocess

import babyai
import babyai.utils as utils

from babyai_sr.rl.algos.ppo import PPOAlgo
from babyai_sr.arguments import ArgumentParser
from babyai_sr.model import ACModel

import babyai_sr.levels.sr_levels

# Parse arguments.
parser = ArgumentParser()
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--reward-scale", type=float, default=20.,
                    help="Reward scale multiplier")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
parser.add_argument("--sender", default=None,
                    help="name of the sender (default: ENV_ALGO_TIME)")
parser.add_argument("--receiver", default=None,
                    help="name of the receiver (default: ENV_ALGO_TIME)")
parser.add_argument("--pretrained-sender", default=None,
                    help="If you\'re using a pre-trained sender and want the fine-tuned one to have a new name")
parser.add_argument("--pretrained-receiver", default=None,
                    help="If you\'re using a pre-trained receiver and want the fine-tuned one to have a new name")
parser.add_argument("--len-message", type=int, default=8,
                    help="lengths of messages (default: 8)")
parser.add_argument("--num-symbols", type=int, default=8,
                    help="number of symbols (default: 8)")
parser.add_argument("--n", type=int, default=64,
                    help="communication interval (default: 64)")
parser.add_argument("--no-comm", action="store_true", default=False,
                    help="don't use communication")
parser.add_argument("--all-angles", action="store_true", default=True,
                    help="let the sender observe the environment from all angles")
parser.add_argument("--archimedean", action="store_true", default=False,
                    help="use Archimedean receiver")
parser.add_argument("--ignorant-sender", action="store_true", default=False,
                    help="blinds the sender to the instruction")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="use argmax for messages instead of sampling")
args = parser.parse_args()

utils.seed(args.seed)

# Generate environments.
envs = []
for i in range(args.procs):
    env = gym.make(args.env)
    env.seed(100 * args.seed + i)
    envs.append(env)

# Define model names.
suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
model_name_parts = {
    'env':    args.env,
    'algo':   "ppo",
    'arch':   "expert_filmcnn",
    'instr':  "gru",
    'mem':    "mem",
    'seed':   args.seed,
    'info':   "",
    'coef':   "",
    'suffix': suffix}

sender_name_parts         = model_name_parts.copy()
sender_name_parts["info"] = "_n%d%s%s_sender" % (args.n, "_no-comm" if args.no_comm else "", "_archimedean" if args.archimedean else "")
default_sender_name       = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**sender_name_parts)
if args.pretrained_sender:
    default_sender_name = args.pretrained_sender + '_pretrained_' + default_sender_name
args.sender = args.sender.format(**sender_name_parts) if args.sender else default_sender_name

receiver_name_parts         = model_name_parts.copy()
receiver_name_parts["info"] = "_n%d%s%s_receiver" % (args.n, "_no-comm" if args.no_comm else "", "_archimedean" if args.archimedean else "")
default_receiver_name       = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**receiver_name_parts)
if args.pretrained_receiver:
    default_receiver_name = args.pretrained_receiver + '_pretrained_' + default_receiver_name
args.receiver = args.receiver.format(**receiver_name_parts) if args.receiver else default_receiver_name

utils.configure_logging(args.receiver)
logger = logging.getLogger(__name__)

# Define obss preprocessor.
obss_preprocessor = utils.ObssPreprocessor(args.receiver, envs[0].observation_space, args.pretrained_receiver)

# Define actor--critic models.
sender = utils.load_model(args.sender, raise_not_found=False)
if sender is None:
    if args.pretrained_sender:
        sender = utils.load_model(args.pretrained_sender, raise_not_found=True)
    else:
        sender = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                         args.image_dim, args.memory_dim, args.instr_dim, args.enc_dim, args.dec_dim,
                         args.len_message, args.num_symbols, args.all_angles)

receiver = utils.load_model(args.receiver, raise_not_found=False)
if receiver is None:
    if args.pretrained_receiver:
        receiver = utils.load_model(args.pretrained_receiver, raise_not_found=True)
    else:
        receiver = ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                           args.image_dim, args.memory_dim, args.instr_dim, args.enc_dim, args.dec_dim,
                           args.len_message, args.num_symbols)

obss_preprocessor.vocab.save()
utils.save_model(sender,   args.sender  )
utils.save_model(receiver, args.receiver)

if torch.cuda.is_available():
    sender.cuda()
    receiver.cuda()

# Define actor--critic algorithm.
reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
algo = PPOAlgo(envs, [sender, receiver], args.n, args.frames_per_proc, args.discount, args.lr, args.beta1,
               args.beta2, args.gae_lambda, args.entropy_coef, args.value_loss_coef,
               args.max_grad_norm, args.recurrence, args.optim_eps, args.clip_eps, args.ppo_epochs,
               args.batch_size, obss_preprocessor, reshape_reward, not args.no_comm, args.archimedean, args.argmax, args.ignorant_sender)

# Restore training status.
status_path = os.path.join(utils.get_log_dir(args.receiver), 'status.json')
if os.path.exists(status_path):
    with open(status_path, 'r') as src:
        status = json.load(src)
else:
    status = {'i':            0,
              'num_episodes': 0,
              'num_frames':   0
             }

# Define logger and Tensorboard writer and CSV writer.
header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])
if args.tb:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(utils.get_log_dir(args.receiver))
csv_path = os.path.join(utils.get_log_dir(args.receiver), 'log.csv')
first_created = not os.path.exists(csv_path)
# we don't buffer data going in the csv log, cause we assume
# that one update will take much longer that one write to the log
csv_writer = csv.writer(open(csv_path, 'a', 1))
if first_created:
    csv_writer.writerow(header)

# Log code state, command, availability of CUDA and models.
babyai_code = list(babyai.__path__)[0]
try:
    last_commit = subprocess.check_output(
        'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
    logger.info('LAST COMMIT INFO:')
    logger.info(last_commit)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
try:
    diff = subprocess.check_output(
        'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
    if diff:
        logger.info('GIT DIFF:')
        logger.info(diff)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
logger.info('COMMAND LINE ARGS:')
logger.info(args)
logger.info("CUDA available: {}".format(torch.cuda.is_available()))
logger.info(sender)
logger.info(receiver)

# Train models.
total_start_time = time.time()
while status["num_frames"] < args.frames:
    # Update parameters
    update_start_time = time.time()
    logs = algo.update_parameters()
    update_end_time = time.time()

    status['num_frames']   += logs["num_frames"]
    status['num_episodes'] += logs['episodes_done']
    status['i']            += 1

    # Print logs.

    if status['i'] % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = datetime.timedelta(seconds=total_ellapsed_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        data = [status['i'], status['num_episodes'], status['num_frames'],
                fps, total_ellapsed_time,
                *return_per_episode.values(),
                success_per_episode['mean'],
                *num_frames_per_episode.values(),
                logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                logs["loss"], logs["grad_norm"]]

        format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

        logger.info(format_str.format(*data))
        if args.tb:
            assert len(header) == len(data)
            for key, value in zip(header, data):
                writer.add_scalar(key, float(value), status['num_frames'])

        csv_writer.writerow(data)
        
    # Save obss preprocessor vocabulary and models.
    if args.save_interval > 0 and status['i'] % args.save_interval == 0:
        obss_preprocessor.vocab.save()
        with open(status_path, 'w') as dst:
            json.dump(status, dst)
            utils.save_model(sender,   args.sender  )
            utils.save_model(receiver, args.receiver)
