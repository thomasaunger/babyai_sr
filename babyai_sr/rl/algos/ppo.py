import numpy as np
import torch

from babyai_sr.rl.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    def __init__(self, env, models, num_frames_per_proc=160, discount=0.99, lr=1e-4, beta1=0.9,
                 beta2=0.999, gae_lambda=0.99, entropy_coef=0.01, value_loss_coef=0.5,
                 max_grad_norm=0.5, recurrence=160, adam_eps=1e-5, clip_eps=0.2, epochs=4,
                 batch_size=5120, preprocess_obss=None, reshape_reward=None, use_comm=True, conventional=False, archimedean=False, argmax=False, ignorant_sender=False):
        
        super().__init__(env, models, num_frames_per_proc, discount, gae_lambda, preprocess_obss, reshape_reward, use_comm, conventional, archimedean, argmax, ignorant_sender)
        
        self.lr              = lr
        self.entropy_coef    = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm   = max_grad_norm
        self.recurrence      = recurrence
        self.clip_eps        = clip_eps
        self.epochs          = epochs
        self.batch_size      = batch_size
        
        assert self.num_frames_per_proc % self.recurrence == 0
        
        assert self.batch_size % self.recurrence == 0
        
        self.optimizers = [torch.optim.Adam(model.parameters(), lr, (beta1, beta2), eps=adam_eps) for model in self.models]
        
        self.batch_num = 0
    
    def update_parameters(self):
        # Collect experiences.
        exps, log = self.collect_experiences()
        
        # Initialize log values.
        log_entropies     = []
        log_values        = []
        log_policy_losses = []
        log_value_losses  = []
        log_grad_norms    = []
        log_losses        = []
        
        for _ in range(self.epochs):
            for inds in self._get_batches_starting_indexes():
                # Initialize batch values.
                batch_entropy     = 0
                batch_value       = 0
                batch_policy_loss = 0
                batch_value_loss  = 0
                batch_loss        = 0
                
                # Initialize.
                value  = torch.zeros(inds.shape[0], self.num_agents, device=self.device)
                memory = exps.memory[inds]
                
                entropies = torch.zeros(inds.shape[0], self.num_agents, device=self.device)
                log_prob  = torch.zeros(inds.shape[0], self.num_agents, device=self.device)
                
                for i in range(self.recurrence):
                    # Create a sub-batch of experience.
                    sb = exps[inds + i]

                    # Compute loss.
                    for m, model in enumerate(self.models):
                        if torch.any(sb.active[:, m]):
                            if m == 1:
                                if self.archimedean:
                                    if self.use_comm:
                                        model_results = model(sb.globs[sb.active[:, m]], memory[sb.active[:, m], m]*sb.mask[sb.active[:, m], m], msg=sb.message[sb.active[:, m], 0])
                                    else:
                                        model_results = model(sb.globs[sb.active[:, m]], memory[sb.active[:, m], m]*sb.mask[sb.active[:, m], m])
                                else:
                                    if self.use_comm:
                                        model_results = model(sb.obs[  sb.active[:, m]], memory[sb.active[:, m], m]*sb.mask[sb.active[:, m], m], msg=sb.message[sb.active[:, m], 0])
                                    else:
                                        model_results = model(sb.obs[  sb.active[:, m]], memory[sb.active[:, m], m]*sb.mask[sb.active[:, m], m])
                            else:
                                if self.ignorant_sender:
                                    sb.globs.instr[sb.active[:, m]] *= 0
                                
                                model_results = model(sb.globs[sb.active[:, m]], memory[sb.active[:, m], m]*sb.mask[sb.active[:, m], m])

                            memory[sb.active[:, m], m] = model_results["memory"]
                            dist                       = model_results["dist"]
                            dists_speaker              = model_results["dists_speaker"]
                            value[ sb.active[:, m], m] = model_results["value"]
                            
                            if m == 1:
                                entropies[sb.active[:, m], m] = dist.entropy()
                                log_prob[ sb.active[:, m], m] = dist.log_prob(sb.action[sb.active[:, m], m])
                            else:
                                entropies[sb.active[:, m], m] = model.speaker_entropy(dists_speaker)
                                log_prob[ sb.active[:, m], m] = model.speaker_log_prob(dists_speaker, sb.message[sb.active[:, m], m])
                    
                    activities = sb.active.sum()
                    
                    entropy = (entropies * sb.active).sum() / activities
                    
                    ratio       =  torch.exp(log_prob - sb.log_prob)
                    surr1       =  ratio * sb.advantage
                    surr2       =  torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -(torch.min(surr1, surr2) * sb.active).sum() / activities
                    
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1         = (value - sb.returnn).pow(2)
                    surr2         = (value_clipped - sb.returnn).pow(2)
                    value_loss    = (torch.max(surr1, surr2) * sb.active).sum() / activities
                    
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                    
                    # Update batch values.
                    batch_entropy     += entropy.item()
                    batch_value       += ((value * sb.active).sum() / activities).item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss  += value_loss.item()
                    batch_loss        += loss
                
                # Update batch values.
                batch_entropy     /= self.recurrence
                batch_value       /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss  /= self.recurrence
                batch_loss        /= self.recurrence
                
                # Update actor--critic.
                [optimizer.zero_grad() for optimizer in self.optimizers]
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for model in self.models for p in model.parameters() if p.grad is not None) ** 0.5
                [torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm) for model in self.models]
                [optimizer.step() for optimizer in self.optimizers]
                
                # Update log values.
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())
    
        # Log some values.
        log["entropy"]     = np.sum(log_entropies)     / (log["num_frames"] * self.epochs)
        log["value"]       = np.sum(log_values)        / (log["num_frames"] * self.epochs)
        log["policy_loss"] = np.sum(log_policy_losses) / (log["num_frames"] * self.epochs)
        log["value_loss"]  = np.sum(log_value_losses)  / (log["num_frames"] * self.epochs)
        log["loss"]        = np.sum(log_losses)        / (log["num_frames"] * self.epochs)
        log["grad_norm"]   = np.mean(log_grad_norms)
        
        return log
    
    def _get_batches_starting_indexes(self):
        indexes = np.arange(0, self.num_frames, self.recurrence)
        indexes = np.random.permutation(indexes)
        
        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]
        
        return batches_starting_indexes
