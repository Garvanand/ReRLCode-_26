import torch
import os
import sys

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.sumo_config import setup_sumo, STATE_DIM, ACTION_DIM
setup_sumo()

from environment.sumo_env import SumoEnv
from traffic_agents.ppo_agent import PPOAgent


def run_demo(model_path=None, steps=3600):
    env_path = os.path.join(project_root, 'environment', '2x2_grid.sumocfg')

    # Check if files exist
    if not os.path.exists(env_path):
        print("SUMO files not found. Generating them now...")
        from environment.generate_grid import generate_network, generate_routes, create_sumocfg
        generate_network()
        generate_routes()
        create_sumocfg()

    # Enable GUI for demo
    env = SumoEnv(env_path, use_gui=True)

    # Init agent with correct dimensions
    agent = PPOAgent(STATE_DIM, ACTION_DIM)

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        agent.policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        agent.policy_old.load_state_dict(agent.policy.state_dict())
        print("Model loaded successfully.")
    else:
        print("Running demo with untrained agent (random actions)...")

    states, _ = env.reset()  # Gymnasium-style 2-tuple
    try:
        for t in range(steps):
            actions = {}
            for jid, state in states.items():
                action, _ = agent.select_action(state)
                actions[jid] = action

            states, rewards, terminated, truncated, info = env.step(actions)

            if terminated or truncated:
                print(f"Simulation finished at step {t}.")
                break
    except KeyboardInterrupt:
        print("Demo interrupted by user.")
    except Exception as e:
        print(f"Demo interrupted: {e}")
    finally:
        env.close()
        print("Demo completed.")


if __name__ == "__main__":
    # Default to the latest trained model
    default_model = os.path.join(project_root, 'models', 'ppo_agent_latest.pth')
    run_demo(default_model)
