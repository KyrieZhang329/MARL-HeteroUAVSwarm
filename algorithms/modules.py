import torch
import torch.nn as nn
import numpy as np
from marl_uav.env.config import OBS_CONFIG, NUM_OBSTACLES


def layer_init(layer,std=np.sqrt(2.0),bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight,std)
    torch.nn.init.constant_(layer.bias,bias_const)
    return layer

class AttentionNetWork(nn.Module):
    def __init__(self,obs_dim,hidden_size,num_agents,num_obstacles,n_head=4):
        super.__init__()

        self.dim_self = OBS_CONFIG['dim_self']
        self.dim_obs_item = OBS_CONFIG['dim_obs_item']
        self.dim_neigh_item = OBS_CONFIG['dim_neigh_item']
        self.num_obs = NUM_OBSTACLES
        self.num_neighbors = num_agents-1
        self.hidden_size = hidden_size

        self.self_embed = nn.Sequential(layer_init(nn.Linear(self.dim_self,hidden_size)),nn.Tanh())
        self.obst_embed = nn.Sequential(layer_init(nn.Linear(self.dim_obs_item,hidden_size)),nn.Tanh())
        self.neig_embed = nn.Sequential(layer_init(nn.Linear(self.dim_neigh_item,hidden_size)),nn.Tanh())
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size,num_heads=n_head,batch_first=True)


    def parse_obs(self, x):
        batch_size = x.shape[0]
        idx_self_end = self.dim_self
        idx_obs_end = idx_self_end+(self.num_obs*self.dim_obs_item)
        self_feat = x[:,:idx_self_end]
        obs_feat = x[:,idx_self_end:idx_obs_end].reshape(batch_size,self.num_obs,self.dim_obs_item)
        neigh_feat = x[:,idx_obs_end:].reshape(batch_size,self.num_neighbors,self.dim_neigh_item)
        return self_feat, obs_feat, neigh_feat


    def forward(self,x):
        self_f,obs_f,neigh_f = self.parse_obs(x)
        q = self.self_embed(self_f).unsqueeze(1)
        k_obs = self.obst_embed(obs_f)
        k_neigh = self.neig_embed(neigh_f)

        k_all = torch.cat([q,k_obs,k_neigh],dim=1)
        attn_output, attn_weights = self.attention(q,k_all,k_all)
        return attn_output.squeeze(1),attn_weights


class PPOActor(nn.Module):
    def __init__(self,obs_dim,action_dim,hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim,hidden_size)),nn.Tanh(),
            layer_init(nn.Linear(hidden_size,hidden_size)),nn.Tanh(),
            layer_init(nn.Linear(hidden_size,action_dim),std=0.01),
        )
        self.logstd = nn.Parameter(torch.zeros(1,action_dim))

    def forward(self,x):
        action_mean = self.net(x)
        action_logstd = self.logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return action_mean,action_std


class SACActor(nn.Module):
    def __init__(self,obs_dim,action_dim,hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim,hidden_size)),nn.Tanh(),
            layer_init(nn.Linear(hidden_size,hidden_size)),nn.Tanh(),
        )
        self.mean_layer = layer_init(nn.Linear(hidden_size,action_dim),std=0.01)
        self.log_std_layer = layer_init(nn.Linear(hidden_size,action_dim),std=0.01)

    def forward(self,x):
        feat = self.net(x)
        mean = self.mean_layer(feat)
        log_std = self.log_std_layer(feat)
        log_std = torch.clamp(log_std,-20,2)
        return mean,log_std

    def get_action(self,x):
        mean,log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean,std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0-y_t.pow(2)+1e-6)
        log_prob = log_prob.sum(1,keepdim=True)
        mean_action = torch.tanh(mean)
        return action,log_prob,mean_action


class QNetwork(nn.Module):
    def __init__(self,obs_dim,action_dim,hidden_size,num_agents,num_obstacles):
        super().__init__()
        self.feature_extractor = AttentionNetWork(obs_dim,action_dim,hidden_size,num_agents,num_obstacles)
        self.q_head = nn.Sequential(
            layer_init(nn.Linear(hidden_size+action_dim,hidden_size)),nn.Tanh(),
            layer_init(nn.Linear(hidden_size,hidden_size)),nn.Tanh(),
            layer_init(nn.Linear(hidden_size,1),std=1.0)
        )

    def forward(self,x,a):
        features,_ =self.feature_extractor(x)
        cat_inputs = torch.cat([features,a],dim=1)
        q_values = self.q_head(cat_inputs)
        return q_values


class Critic(nn.Module):
    def __init__(self,obs_dim,num_agents,hidden_size,num_obs):
        super().__init__()
        self.feature_extractor = AttentionNetWork(obs_dim,hidden_size,num_agents,num_obs)
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(hidden_size,hidden_size)),nn.Tanh(),
            layer_init(nn.Linear(hidden_size,1),std=0.01),
        )

    def forward(self,x):
        features,attn_weights = self.feature_extractor(x)
        value = self.value_head(features)
        return value,attn_weights