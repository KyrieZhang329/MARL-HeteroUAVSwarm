import numpy as np
import torch

class ReplayBuffer:
    def __init__(self,capacity,obs_dim,action_dim,device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity,obs_dim),dtype=np.float32)
        self.actions = np.zeros((capacity,action_dim),dtype=np.float32)
        self.rewards = np.zeros((capacity,1),dtype=np.float32)
        self.next_obs = np.zeros((capacity,obs_dim),dtype=np.float32)
        self.dones = np.zeros((capacity,1),dtype=np.float32)

    def add(self,obs,action,reward,next_obs,done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1)%self.capacity
        self.size = min(self.size+1,self.capacity)

    def sample(self,batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[ind]).to(self.device),
            torch.FloatTensor(self.actions[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.next_obs[ind]).to(self.device),
            torch.FloatTensor(self.dones[ind]).to(self.device)
        )