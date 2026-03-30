import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random
import time
import os

GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
BUFFER_SIZE = 50000
BATCH_SIZE = 64
MAX_EPISODES = 100
MAX_STEPS = 200
WARMUP_STEPS = 500
HIDDEN = 256
SEED = 42

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, n):
        batch = random.sample(self.buffer, n)
        s, a, r, s_, d = zip(*batch)
        s = torch.FloatTensor(np.array(s))
        a = torch.FloatTensor(np.array(a))
        r = torch.FloatTensor(np.array(r)).unsqueeze(1)
        s_ = torch.FloatTensor(np.array(s_))
        d = torch.FloatTensor(np.array(d)).unsqueeze(1)
        return s, a, r, s_, d

    def __len__(self):
        return len(self.buffer)

class OUNoise:
    def __init__(self, dim):
        self.mu = np.zeros(dim)
        self.theta = 0.15
        self.sigma = 0.2
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def __call__(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.mu))
        self.state += dx
        return self.state

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_high):
        super().__init__()
        self.a_high = torch.FloatTensor(a_high)
        self.fc1 = nn.Linear(s_dim, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, a_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.a_high
        return x

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DDPGAgent:
    def __init__(self, s_dim, a_dim, a_high, a_low):
        self.actor = Actor(s_dim, a_dim, a_high)
        self.critic = Critic(s_dim, a_dim)
        self.target_actor = Actor(s_dim, a_dim, a_high)
        self.target_critic = Critic(s_dim, a_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.noise = OUNoise(a_dim)
        self.a_high = torch.FloatTensor(a_high)
        self.a_low = torch.FloatTensor(a_low)

    def _soft_update(self, net, target):
        for param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def act(self, state, explore=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).numpy()
        if explore:
            action += self.noise()
            action = np.clip(action, self.a_low.numpy(), self.a_high.numpy())
        return action

    def remember(self, *args):
        self.buffer.push(*args)

    def learn(self):
        if len(self.buffer) < WARMUP_STEPS:
            return 0.0, 0.0
            
        s, a, r, s_, d = self.buffer.sample(BATCH_SIZE)
        
        with torch.no_grad():
            next_a = self.target_actor(s_)
            y = r + GAMMA * self.target_critic(s_, next_a) * (1 - d)
            
        q_value = self.critic(s, a)
        critic_loss = F.mse_loss(q_value, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(s, self.actor(s)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
        
        return actor_loss.item(), critic_loss.item()

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f" INITIALIZING DDPG ENHANCED VIVA EXPERIMENT")
    print(f"{'='*60}\n")
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    env = gym.make("Pendulum-v1")
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    
    agent = DDPGAgent(s_dim, a_dim, env.action_space.high, env.action_space.low)
    
    rewards = []
    actor_losses = []
    critic_losses = []
    best_avg_reward = -float('inf')

    start_time = time.time()
    print(" Starting Training Phase...")
    
    for episode in range(MAX_EPISODES):
        state, _ = env.reset(seed=SEED)
        agent.noise.reset()
        
        ep_reward = 0
        ep_a_loss, ep_c_loss = 0, 0
        loss_steps = 0
        
        for step in range(MAX_STEPS):
            action = agent.act(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, float(done))
            a_loss, c_loss = agent.learn()
            
            if a_loss != 0 and c_loss != 0:
                ep_a_loss += a_loss
                ep_c_loss += c_loss
                loss_steps += 1
                
            state = next_state
            ep_reward += reward
            if done:
                break
                
        rewards.append(ep_reward)
        if loss_steps > 0:
            actor_losses.append(ep_a_loss / loss_steps)
            critic_losses.append(ep_c_loss / loss_steps)
        else:
            actor_losses.append(0)
            critic_losses.append(0)
            
        if (episode + 1) % 10 == 0:
            avg_10 = np.mean(rewards[-10:])
            print(f"Episode {episode+1:>3}/{MAX_EPISODES} | Reward: {ep_reward:>8.2f} | 10-Ep Avg: {avg_10:>8.2f} | A-Loss: {actor_losses[-1]:>7.3f} | C-Loss: {critic_losses[-1]:>7.3f}")
            
            if avg_10 > best_avg_reward and episode > 10:
                best_avg_reward = avg_10
                torch.save(agent.actor.state_dict(), "best_actor.pth")
                torch.save(agent.critic.state_dict(), "best_critic.pth")

    train_time = time.time() - start_time
    print(f"\n Training Complete in {train_time:.1f}s.")
    print(f" Checkpoints saved: best_actor.pth, best_critic.pth")
    env.close()

    print(f"\n Generating 3-Panel Metrics Dashboard...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    rolling = [np.mean(rewards[max(0, i-9):i+1]) for i in range(len(rewards))]
    ax1.plot(rewards, color="#90CAF9", alpha=0.5, label="Raw Reward")
    ax1.plot(rolling, color="#1565C0", linewidth=2.0, label="10-Ep Rolling Avg")
    ax1.axhline(-200.0, color="#E53935", linestyle="--", alpha=0.5, label="Optimal Target (-200)")
    ax1.set_title("DDPG Training Performance Dashboard", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(critic_losses, color="#FF9800", linewidth=1.5, label="Critic Loss (TD Error)")
    ax2.set_ylabel("MSE Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(actor_losses, color="#4CAF50", linewidth=1.5, label="Actor Loss (Policy Gradient)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ddpg_enhanced_metrics.png", dpi=150)
    print(" Saved enhanced plot to 'ddpg_enhanced_metrics.png'")
    
    print(f"\n{'='*60}")
    print(f" COMMENCING LIVE VISUAL DEMONSTRATION")
    print(f"{'='*60}\n")
    
    try:
        demo_env = gym.make("Pendulum-v1", render_mode="human")
        
        if os.path.exists("best_actor.pth"):
            agent.actor.load_state_dict(torch.load("best_actor.pth", weights_only=True))
            print("Successfully loaded 'best_actor.pth' for demonstration.")
            
        print("\nNow rendering 3 episodes representing the final intelligent agent...")
        for ep in range(3):
            state, _ = demo_env.reset()
            ep_reward = 0
            for step in range(MAX_STEPS):
                action = agent.act(state, explore=False)
                state, reward, terminated, truncated, _ = demo_env.step(action)
                demo_env.render()
                time.sleep(0.01)
                ep_reward += reward
                if terminated or truncated:
                    break
            print(f" [Demo Window] Episode {ep+1}/3 | Real-world score: {ep_reward:.2f}")
            
        demo_env.close()
        print("\n Demonstration Complete! Let's conquer the Viva.")
        
    except Exception as e:
        print(f"\n Could not launch the popup physical simulation rendering window.")
        print(f"Reason: {e}")
        print("Note: To see the live visual popup, make sure you install pygame:")
        print("pip install pygame\n")
