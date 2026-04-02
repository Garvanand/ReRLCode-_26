import torch
import numpy as np
import os
import sys

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.sumo_config import (
    setup_sumo, STATE_DIM, ACTION_DIM,
    DEFAULT_MAX_EPISODES, DEFAULT_MAX_STEPS, UPDATE_TIMESTEP
)
setup_sumo()

from torch.utils.tensorboard import SummaryWriter
from traffic_agents.ppo_agent import PPOAgent, Memory
from environment.sumo_env import SumoEnv

def train():
    # Parameters
    env_path = os.path.join(project_root, 'environment', '2x2_grid.sumocfg')
    
    # Check if SUMO files exist
    if not os.path.exists(env_path):
        from environment.generate_grid import generate_network, generate_routes, create_sumocfg
        generate_network()
        generate_routes()
        create_sumocfg()
        
    state_dim = STATE_DIM    # 12 queue + 12 wait + 12 occupancy + 1 phase
    action_dim = ACTION_DIM  # Switch or Keep
    max_episodes = DEFAULT_MAX_EPISODES
    max_steps = DEFAULT_MAX_STEPS
    update_timestep = UPDATE_TIMESTEP
    log_interval = 1
    save_interval = 20  # save numbered checkpoint every N episodes
    
    # Init env and agent
    env = SumoEnv(env_path, step_length=5, yellow_duration=3)
    shared_agent = PPOAgent(state_dim, action_dim)
    memory = Memory()
    
    # TensorBoard writer
    log_dir = os.path.join(project_root, 'logs')
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    writer = SummaryWriter(os.path.join(log_dir, 'ppo_traffic'))
    
    models_dir = os.path.join(project_root, 'models')
    if not os.path.exists(models_dir): os.makedirs(models_dir)
    
    time_step = 0
    for episode in range(1, max_episodes + 1):
        states, _ = env.reset() # states: {jid: state_vector}
        episode_reward = 0
        
        for t in range(max_steps):
            actions = {}
            logprobs = {}
            
            # Select action for each junction using shared policy
            for jid, state in states.items():
                action, logprob = shared_agent.select_action(state)
                actions[jid] = action
                logprobs[jid] = logprob
                
            # Step environment (Gymnasium style)
            next_states, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            
            # Store in shared memory (Centralized training)
            for jid in states.keys():
                memory.states.append(states[jid])
                memory.actions.append(actions[jid])
                memory.logprobs.append(logprobs[jid])
                memory.rewards.append(rewards[jid])
                memory.is_terminals.append(done)
            
            states = next_states
            episode_reward += sum(rewards.values())
            time_step += len(states) # Number of experiences
            
            # Update PPO
            if time_step >= update_timestep:
                shared_agent.update(memory)
                memory.clear()
                time_step = 0
            
            if done:
                break
        
        env.close()
        
        # Log episode metrics
        avg_reward = episode_reward / len(env.junction_ids)
        writer.add_scalar('Reward/AveragePerJunction', avg_reward, episode)
        
        if episode % log_interval == 0:
            print(f"Episode {episode}/{max_episodes} | Avg Reward: {avg_reward:.2f}")
            
        # Save model
        torch.save(shared_agent.policy.state_dict(),
                   os.path.join(models_dir, 'ppo_agent_latest.pth'))
        # Numbered checkpoint
        if episode % save_interval == 0:
            torch.save(shared_agent.policy.state_dict(),
                       os.path.join(models_dir, f'ppo_agent_{episode}.pth'))
            
    env.close()
    writer.close()

if __name__ == "__main__":
    train()
