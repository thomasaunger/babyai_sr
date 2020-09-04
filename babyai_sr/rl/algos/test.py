import torch

from babyai_sr.rl.algos.base import BaseAlgo

class TestAlgo(BaseAlgo):
    def __init__(self, env, models, num_frames_per_proc=40, discount=0.99, gae_lambda=0.99, preprocess_obss=None, reshape_reward=None, use_comm=True, archimedean=False, argmax=True, ignorant_sender=False):
        
        super().__init__(env, models, num_frames_per_proc, discount, gae_lambda, preprocess_obss, reshape_reward, use_comm, archimedean, argmax, ignorant_sender)
    
    def collect_episodes(self, episodes):
        # Collect experiences.
        exps, _ = self.collect_experiences()
        batch   = 1
        
        active  = exps.active.view( self.num_procs, self.num_frames_per_proc, *exps.active.shape[ 1:])
        extra   = exps.extra.view(  self.num_procs, self.num_frames_per_proc, *exps.extra.shape[  1:])
        mask    = exps.mask.view(   self.num_procs, self.num_frames_per_proc, *exps.mask.shape[   1:])
        done    = exps.done.view(   self.num_procs, self.num_frames_per_proc, *exps.done.shape[   1:])
        message = exps.message.view(self.num_procs, self.num_frames_per_proc, *exps.message.shape[1:])
        action  = exps.action.view( self.num_procs, self.num_frames_per_proc, *exps.action.shape[ 1:])
        reward  = exps.reward.view( self.num_procs, self.num_frames_per_proc, *exps.reward.shape[ 1:])
        
        log = {
            "return_per_episode":     [],
            "num_frames_per_episode": [],
            "num_frames":             0,
            "episodes_done":          0,
        }
        
        exp_data = torch.zeros(self.num_frames, 4 + self.models[0].len_msg + 3 + 5, dtype=torch.uint8)
        
        SENDER   = 0
        RECEIVER = 1
        
        t             = 0
        proc          = 0
        frame         = [0]*self.num_procs
        episode_frame = 0
        episodes_done = 0
        while True:
            if active[proc, frame[proc], RECEIVER]:
                exp_data[t,                          0:4                         ] = extra[  proc, frame[proc],        0:4]
                exp_data[t,                          4:4+self.models[0].len_msg  ] = message[proc, frame[proc], SENDER  ].argmax(-1)
                exp_data[t, 4+self.models[0].len_msg  ]                            = action[ proc, frame[proc], RECEIVER]
                exp_data[t, 4+self.models[0].len_msg+1]                            = ~mask[  proc, frame[proc], RECEIVER]
                exp_data[t, 4+self.models[0].len_msg+2]                            = reward[ proc, frame[proc], RECEIVER].ceil()
                exp_data[t, 4+self.models[0].len_msg+3:4+self.models[0].len_msg+5] = extra[  proc, frame[proc],        4:6]
                exp_data[t, 4+self.models[0].len_msg+5:4+self.models[0].len_msg+7] = extra[  proc, frame[proc],        2:4]
                
                t             += 1
                episode_frame += 1
            
            if done[proc, frame[proc]]:
                episodes_done += 1
                log["return_per_episode"].append(reward[proc, frame[proc], RECEIVER].item())
                log["num_frames_per_episode"].append(episode_frame)
                episode_frame = 0
                if episodes_done == episodes:
                    break
                frame[proc] += 1
                proc         = (proc + 1) % self.num_procs
            else:
                frame[proc] += 1
            
            if t == exp_data.shape[0]:
                exp_data = torch.cat((exp_data, torch.zeros(exp_data.shape, dtype=torch.uint8)), 0)
            
            if frame[proc] == batch*self.num_frames_per_proc:
                exps, _  = self.collect_experiences()
                batch   += 1
                
                next_active  = exps.active.view( self.num_procs, self.num_frames_per_proc, *exps.active.shape[ 1:])
                next_extra   = exps.extra.view(  self.num_procs, self.num_frames_per_proc, *exps.extra.shape[  1:])
                next_mask    = exps.mask.view(   self.num_procs, self.num_frames_per_proc, *exps.mask.shape[   1:])
                next_done    = exps.done.view(   self.num_procs, self.num_frames_per_proc, *exps.done.shape[   1:])
                next_message = exps.message.view(self.num_procs, self.num_frames_per_proc, *exps.message.shape[1:])
                next_action  = exps.action.view( self.num_procs, self.num_frames_per_proc, *exps.action.shape[ 1:])
                next_reward  = exps.reward.view( self.num_procs, self.num_frames_per_proc, *exps.reward.shape[ 1:])
                
                active  = torch.cat((active,  next_active ), 1)
                extra   = torch.cat((extra,   next_extra  ), 1)
                mask    = torch.cat((mask,    next_mask   ), 1)
                done    = torch.cat((done,    next_done   ), 1)
                message = torch.cat((message, next_message), 1)
                action  = torch.cat((action,  next_action ), 1)
                reward  = torch.cat((reward,  next_reward ), 1)
        
        log["num_frames"]    = t
        log["episodes_done"] = episodes_done
        
        return exp_data[:t], log
