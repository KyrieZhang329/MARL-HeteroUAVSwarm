import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from marl_uav.algorithms.modules import PPOActor,Critic

class MAPPOAgent:
    def __init__(self,obs_dim,action_dim,num_agents,num_obs,args,device):
        self.device = device
        self.args = args

        self.actor = PPOActor(obs_dim,action_dim,args.hidden_size).to(device)
        self.critic = Critic(obs_dim,num_agents,args.hidden_size,num_obs).to(device)
        self.optimizer = optim.Adam([
            {'params':self.actor.parameters(),'lr':args.learning_rate},
            {'params':self.critic.parameters(),'lr':args.learning_rate},
        ],eps=1e-5)
        self.last_attn_weights = None

    def get_action_values(self,x,action=None):
        action_mean,action_std = self.actor(x)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        value,attn_weights = self.critic(x)
        self.last_attn_weights = attn_weights.detach()
        return action,probs.log_prob(action).sum(1),probs.entropy().sum(1),value

    def get_value(self,x,action=None):
        value,_ = self.critic(x)
        return value

    def update(self,b_obs,b_actions,b_logprobs,b_returns,b_advantages):
        batch_size = b_obs.size(0)
        minibatch_size = 64
        ppo_epochs = 10
        total_v_loss = 0
        total_pg_loss = 0

        for epoch in range(ppo_epochs):
            indices = torch.randperm(batch_size)
            for start in range(0,batch_size,minibatch_size):
                end =  start+minibatch_size
                mb_inds = indices[start:end]

                _,newlogprob,entropy,newvalue = self.get_action_values(b_obs[mb_inds],b_actions[mb_inds])
                logratio = newlogprob-b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages-mb_advantages.mean())/(mb_advantages.std()+1e-5)
                pg_loss1 = -mb_advantages*ratio
                pg_loss2 = -mb_advantages*torch.clamp(ratio,1-self.args.clip_coef,1+self.args.clip_coef)
                pg_loss = torch.max(pg_loss2,pg_loss1).mean()

                v_loss = 0.5*((newvalue.view(-1)-b_returns[mb_inds])**2).mean()
                loss = pg_loss-self.args.ent_coef*entropy.mean()+v_loss*self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),self.args.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(),self.args.max_grad_norm)
                self.optimizer.step()
                total_v_loss += v_loss.item()
                total_pg_loss += pg_loss.item()

        num_updates = ppo_epochs*(batch_size/minibatch_size)
        return total_v_loss/num_updates,total_pg_loss/num_updates
