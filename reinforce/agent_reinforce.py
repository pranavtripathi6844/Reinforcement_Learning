import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import time


def discount_rewards(r, gamma):
    """
    Compute discounted rewards.
    Works for both (T,) and (T, N_ENVS) tensors.
    """
    discounted_r = torch.zeros_like(r)
    running_add = 0
    if r.dim() > 1:
        running_add = torch.zeros(r.size(-1)).to(r.device)
        
    for t in reversed(range(0, r.size(0))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)
        
        return normal_dist


class Agent(object):
    def __init__(self, policy, device='cpu', baseline=20.0, lr=1e-3):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.baseline = baseline

        self.gamma = 0.99
        self.states = []
        self.action_log_probs = []
        self.rewards = []
        self.dones = []
        
        # For logging
        self.start_time = time.time()

    def update_policy(self):
        if not self.action_log_probs:
            return 0
            
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device)
        dones = torch.stack(self.dones, dim=0).to(self.train_device)

        self.states, self.action_log_probs, self.rewards, self.dones = [], [], [], []

        # Compute discounted returns
        returns = discount_rewards(rewards, self.gamma)
        
        # If baseline is 0, we use standard batch-mean normalization (which is a baseline)
        # If baseline is non-zero, we subtract it from raw returns.
        if self.baseline != 0:
            returns = returns - self.baseline
            # Normalize only the scale (std) to keep the baseline effect on the mean
            returns = returns / (returns.std() + 1e-8)
        else:
            # TRUE No Baseline: Use raw returns (possibly scaled for numerical stability, but NOT centered)
            # The user requested to remove the "returns - returns.mean()" part.
            # We keep division by std to prevent gradient explosion, but do NOT subtract mean.
            returns = returns / (returns.std() + 1e-8)
        
        # Compute policy gradient loss
        policy_loss = -(action_log_probs * returns).mean()
        
        # Update the policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item()

    def get_action(self, state, evaluation=False):
        if isinstance(state, np.ndarray):
            x = torch.from_numpy(state).float().to(self.train_device)
        else:
            x = state.float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:
            return normal_dist.mean, None
        else:
            action = normal_dist.sample()
            # Sum log probs over action dimensions
            action_log_prob = normal_dist.log_prob(action).sum(dim=-1)
            return action, action_log_prob

    def store_outcome(self, action_log_prob, reward, done):
        # We only need log_probs, rewards and dones for REINFORCE
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.FloatTensor(reward).to(self.train_device))
        self.dones.append(torch.FloatTensor(done).to(self.train_device))

