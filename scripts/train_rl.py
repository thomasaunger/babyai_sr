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
import babyai.utils    as utils

import babyai_sr
import babyai_sr.utils as utils_sr

from babyai_sr.arguments import ArgumentParser
from babyai_sr.model     import ACModel
from babyai_sr.rl.algos  import PPOAlgo
from babyai_sr.rl.utils  import ParallelEnv

# Parse arguments.
parser = ArgumentParser()

# Base arguments
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--buffer", action="store_true", default=False,
                    help="buffer logs before saving")

# Model parameters
parser.add_argument("--sender", default=None,
                    help="name of the sender (default: ENV_ALGO_TIME)")
parser.add_argument("--receiver", default=None,
                    help="name of the receiver (default: ENV_ALGO_TIME)")
parser.add_argument("--pretrained-sender", default=None,
                    help="If you\'re using a pre-trained sender and want the fine-tuned one to have a new name")
parser.add_argument("--pretrained-receiver", default=None,
                    help="If you\'re using a pre-trained receiver and want the fine-tuned one to have a new name")
parser.add_argument("--image-dim", type=int, default=512,
                    help="dimensionality of the image embedding")
parser.add_argument("--memory-dim", type=int, default=1024,
                    help="dimensionality of the memory LSTM")
parser.add_argument("--instr-dim", type=int, default=512,
                    help="dimensionality of the memory LSTM")
parser.add_argument("--enc-dim", type=int, default=512,
                    help="dimensionality of the encoder LSTM")
parser.add_argument("--dec-dim", type=int, default=512,
                    help="dimensionality of the decoder LSTM")
parser.add_argument("--len-message", type=int, default=8,
                    help="lengths of messages (default: 8)")
parser.add_argument("--num-symbols", type=int, default=8,
                    help="number of symbols (default: 8)")
parser.add_argument("--single-angle", action="store_true", default=False,
                    help="let the sender observe the environment from a single angle")

# Training arguments
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--frames", type=int, default=int(9e10),
                    help="number of frames of training (default: 9e10)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate (default: 1e-4)")
parser.add_argument("--beta1", type=float, default=0.9,
                    help="beta1 for Adam (default: 0.9)")
parser.add_argument("--beta2", type=float, default=0.999,
                    help="beta2 for Adam (default: 0.999)")
parser.add_argument("--recurrence", type=int, default=160,
                    help="number of timesteps gradient is backpropagated (default: 160)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--batch-size", type=int, default=5120,
                    help="batch size for PPO (default: 5120)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
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

args = parser.parse_args()

utils.seed(args.seed)

# Generate environments.
envs = []
for i in range(args.procs):
    env = gym.make(args.env)
    env.seed(100 * args.seed + i)
    envs.append(env)

penv = ParallelEnv(envs, args.n, args.conventional)

# Define model names.
suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
model_name_parts = {
    "env":    args.env,
    "algo":   "ppo",
    "arch":   "expert_filmcnn",
    "instr":  "gru",
    "mem":    "mem",
    "seed":   args.seed,
    "info":   "",
    "coef":   "",
    "suffix": suffix}

sender_name_parts         = model_name_parts.copy()
sender_name_parts["info"] = "_n%d%s%s%s%s_sender" % (args.n, "_no-comm" if args.no_comm else "",
                                                     "_conventional" if args.conventional else "",
                                                     "_informed" if args.informed_sender else "",
                                                     "_archimedean" if args.archimedean else "")
default_sender_name       = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**sender_name_parts)
if args.pretrained_sender:
    default_sender_name = args.pretrained_sender + "_pretrained_" + default_sender_name
args.sender = args.sender.format(**sender_name_parts) if args.sender else default_sender_name

receiver_name_parts         = model_name_parts.copy()
receiver_name_parts["info"] = "_n%d%s%s%s%s_receiver" % (args.n, "_no-comm" if args.no_comm else "",
                                                         "_conventional" if args.conventional else "",
                                                         "_informed" if args.informed_sender else "",
                                                         "_archimedean" if args.archimedean else "")
default_receiver_name       = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**receiver_name_parts)
if args.pretrained_receiver:
    default_receiver_name = args.pretrained_receiver + "_pretrained_" + default_receiver_name
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
                         args.len_message, args.num_symbols, not args.single_angle)

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
algo = PPOAlgo(penv, [sender, receiver], args.frames_per_proc, args.discount, args.lr, args.beta1,
               args.beta2, args.gae_lambda, args.entropy_coef, args.value_loss_coef,
               args.max_grad_norm, args.recurrence, args.optim_eps, args.clip_eps, args.ppo_epochs,
               args.batch_size, obss_preprocessor, reshape_reward, not args.no_comm, args.conventional, args.archimedean, args.informed_sender)

optimizer_sender = utils_sr.load_optimizer(args.sender, raise_not_found=False)
if optimizer_sender is None:
    if args.pretrained_sender:
        algo.optimizers[0].load_state_dict(utils_sr.load_optimizer(args.pretrained_sender, raise_not_found=True).state_dict())
else:
    algo.optimizers[0].load_state_dict(optimizer_sender.state_dict())

optimizer_receiver = utils_sr.load_optimizer(args.receiver, raise_not_found=False)
if optimizer_receiver is None:
    if args.pretrained_receiver:
        algo.optimizers[1].load_state_dict(utils_sr.load_optimizer(args.pretrained_receiver, raise_not_found=True).state_dict())
else:
    algo.optimizers[1].load_state_dict(optimizer_receiver.state_dict())

utils_sr.save_optimizer(algo.optimizers[0], args.sender  )
utils_sr.save_optimizer(algo.optimizers[1], args.receiver)

# Restore training status.
status_path = os.path.join(utils.get_log_dir(args.receiver), "status.json")
if os.path.exists(status_path):
    with open(status_path, 'r') as src:
        status = json.load(src)
else:
    status = {"i":            0,
              "num_episodes": 0,
              "num_frames":   0
             }

# Define logger and Tensorboard writer and CSV writer.
header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ["mean", "std", "min", "max"]]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ["mean", "std", "min", "max"]]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])
if args.tb:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(utils.get_log_dir(args.receiver))
csv_path = os.path.join(utils.get_log_dir(args.receiver), "log.csv")
first_created = not os.path.exists(csv_path)
# we don't buffer data going in the csv log, cause we assume
# that one update will take much longer that one write to the log
csv_writer = csv.writer(open(csv_path, 'a', 1))
if first_created:
    csv_writer.writerow(header)

# Log code state, command, availability of CUDA and models.
babyai_code = list(babyai_sr.__path__)[0]
try:
    last_commit = subprocess.check_output(
        "cd {}; git log -n1".format(babyai_code), shell=True).decode("utf-8")
    logger.info("LAST COMMIT INFO:")
    logger.info(last_commit)
except subprocess.CalledProcessError:
    logger.info("Could not figure out the last commit")
try:
    diff = subprocess.check_output(
        "cd {}; git diff".format(babyai_code), shell=True).decode("utf-8")
    if diff:
        logger.info("GIT DIFF:")
        logger.info(diff)
except subprocess.CalledProcessError:
    logger.info("Could not figure out the last commit")
logger.info("COMMAND LINE ARGS:")
logger.info(args)
logger.info("CUDA available: {}".format(torch.cuda.is_available()))
logger.info(sender)
logger.info(receiver)

datas = []

# Train models.
sender.train()
receiver.train()
total_start_time = time.time()
while status["num_frames"] < args.frames:
    # Update parameters
    update_start_time = time.time()
    logs = algo.update_parameters()
    update_end_time = time.time()

    status["num_frames"]   += logs["num_frames"]
    status["num_episodes"] += logs["episodes_done"]
    status["i"]            += 1

    # Print logs.
    
    format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

    if status["i"] % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = datetime.timedelta(seconds=total_ellapsed_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        data = [status["i"], status["num_episodes"], status["num_frames"],
                fps, total_ellapsed_time,
                *return_per_episode.values(),
                success_per_episode["mean"],
                *num_frames_per_episode.values(),
                logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                logs["loss"], logs["grad_norm"]]
        
        datas.append(data)
        
        if not args.buffer:
            logger.info(format_str.format(*data))
            if args.tb:
                assert len(header) == len(data)
                for key, value in zip(header, data):
                    writer.add_scalar(key, float(value), status["num_frames"])
        
            csv_writer.writerow(data)
    
    # Save obss preprocessor vocabulary, buffered logs, models and optimizers.
    if args.save_interval > 0 and status["i"] % args.save_interval == 0:
        obss_preprocessor.vocab.save()
        
        if args.buffer:
            logger.info("\n" + "\n".join([format_str.format(*data) for data in datas]))
            if args.tb:
                for data in datas:
                    assert len(header) == len(data)
                    for key, value in zip(header, data):
                        writer.add_scalar(key, float(value), status["num_frames"])
            
            csv_writer.writerows(datas)

        datas.clear()
        
        with open(status_path, 'w') as dst:
            json.dump(status, dst)
            utils.save_model(sender,   args.sender  )
            utils.save_model(receiver, args.receiver)
            utils_sr.save_optimizer(algo.optimizers[0], args.sender  )
            utils_sr.save_optimizer(algo.optimizers[1], args.receiver)
