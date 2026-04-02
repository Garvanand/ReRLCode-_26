import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        from models.actor_critic import ActorCritic
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = torch.nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.policy_old(state)
        
        dist = Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action).item()

    def update(self, memory):
        # Convert list to tensor
        old_states = torch.FloatTensor(np.array(memory.states))
        old_actions = torch.LongTensor(np.array(memory.actions))
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs))
        
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values = self.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*self.entropy(old_states)
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def evaluate(self, state, action):
        action_probs, state_values = self.policy(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        return action_logprobs, torch.squeeze(state_values)

    def entropy(self, state):
        action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        return dist.entropy()

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
