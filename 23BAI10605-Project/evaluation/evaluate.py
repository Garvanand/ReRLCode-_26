import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.sumo_config import setup_sumo, STATE_DIM, ACTION_DIM
setup_sumo()

import traci
from environment.sumo_env import SumoEnv
from traffic_agents.ppo_agent import PPOAgent

class BaselineAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        
    def select_action(self, state, mode='random'):
        if mode == 'random':
            return np.random.choice(self.action_dim), 0
        elif mode == 'fixed':
            return 0, 0 

def evaluate(model_path=None, use_gui=False, steps=200, live_log=False):
    env_path = os.path.join(project_root, 'environment', '2x2_grid.sumocfg')
    env = SumoEnv(env_path, use_gui=use_gui)
    
    # Init agents
    ppo_agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    if model_path and os.path.exists(model_path):
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        ppo_agent.policy_old.load_state_dict(ppo_agent.policy.state_dict())
    
    baseline_agent = BaselineAgent(action_dim=2)
    
    # Live log file setup
    logs_dir = os.path.join(project_root, 'logs')
    live_log_path = os.path.join(logs_dir, 'live_metrics.csv')
    if live_log:
        if not os.path.exists(logs_dir): os.makedirs(logs_dir)
        # Write header
        with open(live_log_path, 'w') as f:
            f.write("step,avg_wait,avg_queue,throughput\n")

    # Manual action file setup
    manual_action_path = os.path.join(logs_dir, 'manual_actions.json')
    if not os.path.exists(logs_dir): os.makedirs(logs_dir)
    with open(manual_action_path, 'w') as f:
        json.dump({}, f)

    if use_gui:
        print("Starting visual simulation...")
        states, _ = env.reset()
        total_throughput = 0
        for t in range(steps):
            # Check for manual actions or use PPO
            actions = {}
            try:
                with open(manual_action_path, 'r') as f:
                    manual_actions = json.load(f)
            except:
                manual_actions = {}

            for jid, state in states.items():
                if jid in manual_actions:
                    actions[jid] = manual_actions[jid]
                else:
                    action, _ = ppo_agent.select_action(state)
                    actions[jid] = action
            
            # Reset manual actions after read if we want a momentary override
            # Or keep them until user changes back
            
            next_states, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            
            # Log metrics
            if live_log:
                wait = sum([traci.lane.getWaitingTime(l) for j in env.junction_ids for l in traci.trafficlight.getControlledLanes(j)]) / len(env.junction_ids)
                queue = sum([traci.lane.getLastStepHaltingNumber(l) for j in env.junction_ids for l in traci.trafficlight.getControlledLanes(j)]) / len(env.junction_ids)
                total_throughput += traci.simulation.getArrivedNumber()
                
                # Print status every 10 steps to show traffic running
                if t % 10 == 0:
                    veh_count = traci.simulation.getMinExpectedNumber()
                    print(f"[Step {t:4d}] Vehicles in simulation: {veh_count:3d} | Avg Wait: {wait:5.1f}s | Avg Queue: {queue:4.1f} | Throughput: {total_throughput:4d}", flush=True)
                    # Show phase of first 3 junctions to show lights changing
                    phase_str = " | ".join([f"{jid}: {traci.trafficlight.getPhase(jid)}" for jid in env.junction_ids[:3]])
                    print(f"         Phases -> {phase_str}", flush=True)

                with open(live_log_path, 'a') as f:
                    f.write(f"{t},{wait:.2f},{queue:.2f},{total_throughput}\n")

                # Write detailed junction status for dashboard visualization
                junction_data = {}
                for jid in env.junction_ids:
                    lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(jid)))
                    q_len = sum([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
                    phase = traci.trafficlight.getPhase(jid)
                    junction_data[jid] = {"queue": q_len, "phase": int(phase)}
                
                with open(os.path.join(logs_dir, 'junction_states.json'), 'w') as f:
                    json.dump(junction_data, f)


            states = next_states
            if done:
                break
            # Add a small delay for better visualization if GUI is on but no delay in SUMO
            # time.sleep(0.1) 
            
        env.close()
        return None

    results = []
    # Run evaluation for different policies
    for mode in ['PPO', 'Fixed', 'Random']:
        print(f"Running evaluation for {mode}...")
        states, _ = env.reset()
        total_waiting_time = 0
        total_queue_length = 0
        total_throughput = 0
        
        for t in range(steps):
            actions = {}
            for jid, state in states.items():
                if mode == 'PPO':
                    action, _ = ppo_agent.select_action(state)
                elif mode == 'Fixed':
                    action, _ = baseline_agent.select_action(state, mode='fixed')
                else: 
                    action, _ = baseline_agent.select_action(state, mode='random')
                actions[jid] = action
            
            next_states, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            
            wait = sum([traci.lane.getWaitingTime(l) for j in env.junction_ids for l in traci.trafficlight.getControlledLanes(j)])
            total_waiting_time += wait
            queue = sum([traci.lane.getLastStepHaltingNumber(l) for j in env.junction_ids for l in traci.trafficlight.getControlledLanes(j)])
            total_queue_length += queue
            total_throughput += traci.simulation.getArrivedNumber()
            
            states = next_states
            if done:
                break
        
        results.append({
            'Policy': mode,
            'Avg Waiting Time': total_waiting_time / steps / len(env.junction_ids),
            'Avg Queue Length': total_queue_length / steps / len(env.junction_ids),
            'Total Throughput': total_throughput
        })
        env.close()
    
    # Plot results
    df = pd.DataFrame(results)
    df.set_index('Policy', inplace=True)
    plots_dir = os.path.join(project_root, 'plots')
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    df['Avg Waiting Time'].plot(kind='bar', ax=axes[0], title='Avg Waiting Time (Lower is Better)')
    df['Avg Queue Length'].plot(kind='bar', ax=axes[1], title='Avg Queue Length (Lower is Better)')
    df['Total Throughput'].plot(kind='bar', ax=axes[2], title='Total Throughput (Higher is Better)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'comparison_results.png'))
    eval_dir = os.path.join(project_root, 'evaluation')
    df.to_csv(os.path.join(eval_dir, 'evaluation_results.csv'))
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Use SUMO-GUI')
    parser.add_argument('--steps', type=int, default=200, help='Number of simulation steps')
    parser.add_argument('--live', action='store_true', help='Log live metrics')
    args = parser.parse_args()
    
    latest_model = os.path.join(project_root, 'models', 'ppo_agent_latest.pth')
    evaluate(latest_model, use_gui=args.gui, steps=args.steps, live_log=args.live)
