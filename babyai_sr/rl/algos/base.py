import torch

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList

class BaseAlgo():
    def __init__(self, env, models, num_frames_per_proc, discount, gae_lambda, preprocess_obss, reshape_reward, use_comm, conventional, argmax):
        
        # Store parameters.
        self.env                 = env
        self.models              = models
        self.num_frames_per_proc = num_frames_per_proc
        self.discount            = discount
        self.gae_lambda          = gae_lambda
        self.preprocess_obss     = preprocess_obss or default_preprocess_obss
        self.reshape_reward      = reshape_reward
        self.use_comm            = use_comm
        self.conventional        = conventional
        self.argmax              = argmax
        
        # Store helper values.
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agents = len(models)
        self.num_procs  = env.num_procs
        self.num_frames = self.num_frames_per_proc * self.num_procs
        
        # Initialize experience values.
        shape = (self.num_frames_per_proc, self.num_procs, self.num_agents)
        
        self.mask       = torch.zeros(*shape[1:], device=self.device, dtype=torch.bool )
        self.masks      = torch.zeros(*shape,     device=self.device, dtype=torch.bool )
        self.dones      = torch.zeros(*shape[:2], device=self.device, dtype=torch.bool )
        self.activity   = torch.zeros(*shape,     device=self.device, dtype=torch.bool )
        self.actings    = torch.zeros(*shape,     device=self.device, dtype=torch.bool )
        self.sendings   = torch.zeros(*shape,     device=self.device, dtype=torch.bool )
        self.actions    = torch.zeros(*shape,     device=self.device, dtype=torch.uint8)
        self.values     = torch.zeros(*shape,     device=self.device)
        self.rewards    = torch.zeros(*shape,     device=self.device)
        self.advantages = torch.zeros(*shape,     device=self.device)
        self.log_probs  = torch.zeros(*shape,     device=self.device)
        
        self.obss = [None]*self.num_frames_per_proc
        
        self.memory   = torch.zeros(*shape[1:], self.models[0].memory_size, device=self.device)
        self.memories = torch.zeros(*shape,     self.models[0].memory_size, device=self.device)
        
        self.message  = torch.zeros(*shape[1:], self.models[0].len_msg, self.models[0].num_symbols, device=self.device, dtype=torch.float)
        self.messages = torch.zeros(*shape,     self.models[0].len_msg, self.models[0].num_symbols, device=self.device, dtype=torch.float)
        
        active, acting, sending, self.obs, extra = self.env.reset()
        
        self.active  = torch.tensor(active,  device=self.device, dtype=torch.bool)
        self.acting  = torch.tensor(acting,  device=self.device, dtype=torch.bool)
        self.sending = torch.tensor(sending, device=self.device, dtype=torch.bool)
        
        self.extra  = torch.tensor(extra, device=self.device)
        self.extras = torch.zeros(self.num_frames_per_proc, *self.extra.shape, device=self.device)
        
        # Initialize log values.
        self.log_episode_return     = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)
        
        self.log_done_counter = 0
        self.log_return       = [0] * self.num_procs
        self.log_num_frames   = [0] * self.num_procs
    
    def collect_experiences(self):
        # Model inputs.
        memory  = self.memory.clone()
        message = self.message.clone()
        
        # Model outputs.
        action   = torch.zeros(self.num_procs, self.num_agents, device=self.device, dtype=torch.uint8)
        value    = torch.zeros(self.num_procs, self.num_agents, device=self.device)
        log_prob = torch.zeros(self.num_procs, self.num_agents, device=self.device)
        
        for f in range(self.num_frames_per_proc):
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                for m, model in enumerate(self.models):
                    if torch.any(self.active[:, m]):
                        if self.use_comm:
                            model_results = model(preprocessed_obs[self.active[:, m]][m], self.memory[self.active[:, m], m]*self.mask[self.active[:, m], m].unsqueeze(1), msg=self.message[self.active[:, m], 0])
                        else:
                            model_results = model(preprocessed_obs[self.active[:, m]][m], self.memory[self.active[:, m], m]*self.mask[self.active[:, m], m].unsqueeze(1))
                        
                        memory[self.active[:, m], m] = model_results["memory"]
                        dist                         = model_results["dist"]
                        dists_speaker                = model_results["dists_speaker"]
                        value[ self.active[:, m], m] = model_results["value"]
                        
                        action[ self.active[:, m], m] = dist.sample().byte()
                        if self.argmax:
                            message[self.active[:, m]*self.sending[:, m], m] = torch.zeros(message[self.active[:, m], m].size()).scatter(-1, dists_speaker.logits.argmax(-1, keepdim=True), 1)[self.sending[self.active[:, m], m]]
                        else:
                            message[self.active[:, m]*self.sending[:, m], m] = dists_speaker.sample()[self.sending[self.active[:, m], m]]
                        
                        log_prob[self.active[:, m],                    m]  = 0
                        log_prob[self.active[:, m]*self.acting[ :, m], m]  = dist.log_prob(action[self.active[:, m], m])[self.acting[self.active[:, m], m]]
                        log_prob[self.active[:, m]*self.sending[:, m], m] += model.speaker_log_prob(dists_speaker, message[self.active[:, m], m])[self.sending[self.active[:, m], m]]
            
            active, acting, sending, obs, extra, reward, done = self.env.step(action.cpu().numpy())
            
            # Update experience values.
            self.obss[f]     = self.obs
            self.obs         = obs
            
            self.extras[f]   = self.extra
            self.extra       = torch.tensor(extra, device=self.device)
            
            self.memories[f] = self.memory
            self.memory      = memory
            
            self.masks[f]    = self.mask
            self.mask        = ~torch.tensor(done, device=self.device, dtype=torch.bool).unsqueeze(1)*~(~self.mask*~self.active)
            
            self.dones[f]    = torch.tensor(done, device=self.device, dtype=torch.bool)
            
            self.actions[f]  = action
            
            self.messages[f] = self.message
            self.message     = message
            
            self.values[f]   = value
            
            self.actings[f]  = self.acting
            self.acting      = torch.tensor(acting, device=self.device, dtype=torch.bool)
            
            self.sendings[f] = self.sending
            self.sending     = torch.tensor(sending, device=self.device, dtype=torch.bool)
            
            if self.reshape_reward is not None:
                self.rewards[f, :, 1] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action[:, 1], reward, done)
                ], device=self.device)
            else:
                self.rewards[f, :, 1] = torch.tensor(reward, device=self.device)
            
            self.log_probs[f] = log_prob
            
            # Update log values.
            self.log_episode_return     += torch.tensor(reward, device=self.device)
            self.log_episode_num_frames += self.active[:, 1]
            
            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())
            
            self.log_episode_return     *= ~torch.tensor(done, device=self.device, dtype=torch.bool)
            self.log_episode_num_frames *= ~torch.tensor(done, device=self.device, dtype=torch.bool)
            
            # Update activity values.
            self.activity[f] = self.active
            self.active      = torch.tensor(active,  device=self.device, dtype=torch.bool)
        
        # Add advantage and return to experiences.
        next_value = torch.zeros(self.num_procs, self.num_agents, device=self.device)
        
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            for m, model in enumerate(self.models):
                if self.use_comm:
                    model_results = model(preprocessed_obs[:][m], self.memory[:, m]*self.mask[:, m].unsqueeze(1), msg=message[:, 0])
                else:
                    model_results = model(preprocessed_obs[:][m], self.memory[:, m]*self.mask[:, m].unsqueeze(1))
                
                next_value[:, m] = model_results["value"]
                dists_speaker    = model_results["dists_speaker"]
                
                if self.argmax:
                    message[self.active[:, m]*self.sending[:, m], m] = torch.zeros(message[:, m].size()).scatter(-1, dists_speaker.logits.argmax(-1, keepdim=True), 1)[self.active[:, m]*self.sending[:, m]]
                else:
                    message[self.active[:, m]*self.sending[:, m], m] = dists_speaker.sample()[self.active[:, m]*self.sending[:, m]]
        
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask      = self.masks[     i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value     = self.values[    i + 1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else torch.zeros(self.advantages[i].size(), device=self.device)
            
            if self.conventional:
                if i < self.num_frames_per_proc - 1:
                    self.rewards[i, self.activity[i, :, 0], 0] = self.rewards[i + 1, self.activity[i, :, 0], 1]
                
                self.values[    i, self.activity[i, :, 1], 0] = next_value[    self.activity[i, :, 1], 0]
                self.advantages[i, self.activity[i, :, 1], 0] = next_advantage[self.activity[i, :, 1], 0]
            else:
                next_value[    self.activity[i, :, 0], 0] = next_value[    self.activity[i, :, 0], 1]
                next_advantage[self.activity[i, :, 0], 0] = next_advantage[self.activity[i, :, 0], 1]
            
            self.values[    i, self.activity[i, :, 0], 1] = next_value[    self.activity[i, :, 0], 1]
            self.advantages[i, self.activity[i, :, 0], 1] = next_advantage[self.activity[i, :, 0], 1]
            
            delta              = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
        
        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk.
        exps = DictList()
        
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        
        # In commments below, T is self.num_frames_per_proc,
        # P is self.num_procs, M is self.num_agents and D is the dimensionality.
        
        # T x P x D -> P x T x D -> (P * T) x D
        exps.extra = self.extras.transpose(0, 1).reshape(-1, *self.extras.shape[2:])
        exps.done  = self.dones.transpose( 0, 1).reshape(-1, *self.dones.shape[ 2:])
        
        # T x P x M x D -> P x T x M x D -> (P * T) x M x D
        exps.memory  = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        exps.message = self.messages.transpose(0, 1).reshape(-1, *self.messages.shape[2:])
        
        # T x P x M -> P x T x M -> (P * T) x M -> (P * T) x M x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1, *self.masks.shape[2:]).unsqueeze(2)
        
        # For all tensors below, T x P x M -> P x T x M -> (P * T) x M
        exps.active    = self.activity.transpose(  0, 1).reshape(-1, *self.activity.shape[  2:])
        exps.acting    = self.actings.transpose(   0, 1).reshape(-1, *self.actings.shape[  2:])
        exps.sending   = self.sendings.transpose(  0, 1).reshape(-1, *self.sendings.shape[  2:])
        exps.action    = self.actions.transpose(   0, 1).reshape(-1, *self.actions.shape[   2:])
        exps.value     = self.values.transpose(    0, 1).reshape(-1, *self.values.shape[    2:])
        exps.reward    = self.rewards.transpose(   0, 1).reshape(-1, *self.rewards.shape[   2:])
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1, *self.advantages.shape[2:])
        exps.returnn   = exps.value + exps.advantage
        exps.log_prob  = self.log_probs.transpose( 0, 1).reshape(-1, *self.log_probs.shape[ 2:])
        
        # Preprocess experiences.
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        
        # Log some values.
        keep = max(self.log_done_counter, self.num_procs)
        
        log = {
            "return_per_episode":     self.log_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames":             self.num_frames,
            "episodes_done":          self.log_done_counter,
        }
        
        self.log_done_counter = 0
        self.log_return       = self.log_return[-self.num_procs:]
        self.log_num_frames   = self.log_num_frames[-self.num_procs:]

        return exps, log
