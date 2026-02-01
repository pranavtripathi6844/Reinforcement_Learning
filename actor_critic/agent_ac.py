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


class Value(torch.nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3 = torch.nn.Linear(self.hidden, 1)
        self.tanh = torch.nn.Tanh()
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return self.fc3(x)


class ActorCriticAgent(object):
    def __init__(self, policy, value, device='cpu', lr_actor=1e-3, lr_critic=1e-3):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.value = value.to(self.train_device)
        self.optimizer_actor = torch.optim.Adam(policy.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(value.parameters(), lr=lr_critic)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        
        # For logging
        self.start_time = time.time()

    def update(self):
        if not self.action_log_probs:
            return 0, 0
            
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device)
        states = torch.stack(self.states, dim=0).to(self.train_device)
        # next_states not strictly needed for MC AC unless using GAE, but we store them anyway
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device)
        dones = torch.stack(self.done, dim=0).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        # Calculate Monte Carlo discounted returns
        returns = discount_rewards(rewards, self.gamma)

        # Normalize returns (optional but recommended for stability)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Get value estimates for the states visited
        values = self.value(states).squeeze(-1)
        
        # Calculate Advantages (Returns - Baseline)
        # Baseline is the Value function
        advantages = returns - values
        
        # Actor loss
        # Detach advantages to stop gradients flowing into Critic from Actor loss
        actor_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Critic loss (MSE between Value pred and actual Return)
        critic_loss = F.mse_loss(values, returns)
        
        # Update Actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # Update Critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        return actor_loss.item(), critic_loss.item()

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
            action_log_prob = normal_dist.log_prob(action).sum(dim=-1)
            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.FloatTensor(state).to(self.train_device))
        self.next_states.append(torch.FloatTensor(next_state).to(self.train_device))
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.FloatTensor(reward).to(self.train_device))
        self.done.append(torch.FloatTensor(done).to(self.train_device))

