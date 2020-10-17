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

penv = ParallelEnv(envs, args.n, args.conventional, args.archimedean, args.informed_sender)

# Define model names.
roles       = [               "sender",               "receiver"]
model_names = [           args.sender,            args.receiver ]
pretrained  = [args.pretrained_sender, args.pretrained_receiver ]
for m, model_name in enumerate(model_names):
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
    
    model_name_parts["info"] = ("_n%d%s%s%s%s_" + roles[m]) % (args.n, "_no-comm" if args.no_comm else "",
                                                               "_conventional" if args.conventional else "",
                                                               "_informed" if args.informed_sender else "",
                                                               "_archimedean" if args.archimedean else "")
    default_model_name = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{suffix}".format(**model_name_parts)
    if pretrained[m]:
        default_model_name = pretrained[m] + "_pretrained_" + default_model_name
    model_names[m] = model_names[m].format(**model_name_parts) if model_name else default_model_name

loggers = []
for model_name in model_names[:-1]:
    loggers.append(utils_sr.configure_logging(model_name))
loggers.append(utils_sr.configure_logging(model_names[-1], stream=True))

# Define obss preprocessor.
obss_preprocessor = utils_sr.MultiObssPreprocessor(model_names[1], envs[0].observation_space, pretrained[1])

# Define actor--critic models.
models = []
for m, model_name in enumerate(model_names):
    model = utils.load_model(model_name, raise_not_found=False)
    if model is None:
        if pretrained[m]:
            models.append(utils.load_model(pretrained[m], raise_not_found=True))
        else:
            models.append(ACModel(obss_preprocessor.obs_space, envs[0].action_space,
                          args.image_dim, args.memory_dim, args.instr_dim, args.enc_dim, args.dec_dim,
                          args.len_message, args.num_symbols))
    else:
        models.append(model)

obss_preprocessor.vocab.save()
for m, model in enumerate(models):
    utils.save_model(model, model_names[m])

if torch.cuda.is_available():
    for model in models:
        model.cuda()

# Define actor--critic algorithm.
reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
algo = PPOAlgo(penv, models, args.frames_per_proc, args.discount, args.lr, args.beta1,
               args.beta2, args.gae_lambda, args.entropy_coef, args.value_loss_coef,
               args.max_grad_norm, args.recurrence, args.optim_eps, args.clip_eps, args.ppo_epochs,
               args.batch_size, obss_preprocessor, reshape_reward, not args.no_comm, args.conventional)

for m, model_name in enumerate(model_names):
    optimizer = utils_sr.load_optimizer(model_name, raise_not_found=False)
    if optimizer is None:
        if pretrained[m]:
            algo.optimizers[m].load_state_dict(utils_sr.load_optimizer(pretrained[m], raise_not_found=True).state_dict())
    else:
        algo.optimizers[m].load_state_dict(optimizer.state_dict())
    
    utils_sr.save_optimizer(algo.optimizers[m], model_name)

# Restore training status.
status_paths = []
statuses     = []
for m, model_name in enumerate(model_names):
    status_paths.append(os.path.join(utils.get_log_dir(model_name), "status.json"))
    if os.path.exists(status_paths[m]):
        with open(status_paths[m], 'r') as src:
            statuses.append(json.load(src))
    else:
        statuses.append({"i":            0,
                         "num_episodes": 0,
                         "num_frames":   0
                        })

# Define logger and Tensorboard writer and CSV writer.
header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ["mean", "std", "min", "max"]]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ["mean", "std", "min", "max"]]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])
if args.tb:
    from tensorboardX import SummaryWriter
    
    writers = []
    for model_name in model_names:
        writers.append(SummaryWriter(utils.get_log_dir(model_name)))
csv_writers = []
for m, model_name in enumerate(model_names):
    csv_path = os.path.join(utils.get_log_dir(model_name), "log.csv")
    first_created = not os.path.exists(csv_path)
    # we don't buffer data going in the csv log, cause we assume
    # that one update will take much longer than one write to the log
    csv_writers.append(csv.writer(open(csv_path, 'a', 1)))
    if first_created:
        csv_writers[m].writerow(header)

# Log code state, command, availability of CUDA and models.
babyai_code = list(babyai_sr.__path__)[0]
for logger in loggers:
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
    for model in models:
        logger.info(model)

datas = [[] for _ in models]

# Train models.
for model in models:
    model.train()
total_start_time = time.time()
while sum([status["num_frames"] for status in statuses]) < args.frames:
    # Update parameters
    update_start_time = time.time()
    logs = algo.update_parameters()
    update_end_time = time.time()
    
    for m, status in enumerate(statuses):
        status["num_frames"]   += logs[m]["num_frames"]
        status["num_episodes"] += logs[m]["episodes_done"]
        status["i"]            += 1
    
    # Print logs.
    
    format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                  "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                  "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")
    
    if min([status["i"] for status in statuses]) % args.log_interval == 0:
        for m, log in enumerate(logs):
            total_ellapsed_time = int(time.time() - total_start_time)
            fps = log["num_frames"] / (update_end_time - update_start_time)
            duration = datetime.timedelta(seconds=total_ellapsed_time)
            return_per_episode = utils.synthesize(log["return_per_episode"])
            success_per_episode = utils.synthesize(
                [1 if r > 0 else 0 for r in log["return_per_episode"]])
            num_frames_per_episode = utils.synthesize(log["num_frames_per_episode"])
            
            data = [statuses[m]["i"], statuses[m]["num_episodes"], statuses[m]["num_frames"],
                    fps, total_ellapsed_time,
                    *return_per_episode.values(),
                    success_per_episode["mean"],
                    *num_frames_per_episode.values(),
                    log["entropy"], log["value"], log["policy_loss"], log["value_loss"],
                    log["loss"], log["grad_norm"]]
            
            datas[m].append(data)
            
            if not args.buffer:
                loggers[m].info(format_str.format(*data))
                if args.tb:
                    assert len(header) == len(data)
                    for key, value in zip(header, data):
                        writer[m].add_scalar(key, float(value), statuses[m]["num_frames"])
            
                csv_writers[m].writerow(data)
    
    # Save obss preprocessor vocabulary, buffered logs, models and optimizers.
    if args.save_interval > 0 and min([status["i"] for status in statuses]) % args.save_interval == 0:
        obss_preprocessor.vocab.save()
        
        for m, _ in enumerate(logs):
            if args.buffer:
                loggers[m].info("\n" + "\n".join([format_str.format(*data) for data in datas[m]]))
                if args.tb:
                    for data in datas[m]:
                        assert len(header) == len(data)
                        for key, value in zip(header, data):
                            writer[m].add_scalar(key, float(value), statuses[m]["num_frames"])
                
                csv_writers[m].writerows(datas[m])
            
            datas[m].clear()
            
            with open(status_paths[m], 'w') as dst:
                json.dump(statuses[m], dst)
                utils.save_model(            models[    m], model_names[m])
                utils_sr.save_optimizer(algo.optimizers[m], model_names[m])
