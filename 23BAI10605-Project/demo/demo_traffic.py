import os
import sys
import time

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.sumo_config import setup_sumo
setup_sumo()

import traci
from environment.sumo_env import SumoEnv

def demo_run(steps=100):
    env_path = os.path.join(project_root, 'environment', '2x2_grid.sumocfg')
    # Use sumo (no gui) to print out the status to the terminal
    env = SumoEnv(env_path, use_gui=False)
    
    print("Starting textual demo of traffic movement...", flush=True)
    states, _ = env.reset()
    
    # Get all junction IDs
    junctions = env.junction_ids
    if not junctions:
        print("No junctions found!", flush=True)
        return
        
    target_jid = junctions[0]
    print(f"Monitoring Junction: {target_jid}", flush=True)
    
    for t in range(steps):
        # Every 20 steps, we force a phase switch for the target junction to show it changing
        actions = {jid: 0 for jid in junctions}
        if t % 20 == 0 and t > 0:
            actions[target_jid] = 1
            print(f"\n[Step {t}] --- SWITCHING PHASE for {target_jid} ---", flush=True)
            
        states, rewards, terminated, truncated, _ = env.step(actions)
        
        # Get phase state
        phase = traci.trafficlight.getPhase(target_jid)
        ryg = traci.trafficlight.getRedYellowGreenState(target_jid)
        
        # Get vehicles near junction
        lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(target_jid)))
        vehicles = sum([traci.lane.getLastStepVehicleNumber(l) for l in lanes])
        waiting = sum([traci.lane.getWaitingTime(l) for l in lanes])
        
        if t % 5 == 0:
            print(f"[Step {t:3d}] Phase: {phase} | RYG: {ryg} | Vehicles near junction: {vehicles} | Total Wait: {waiting:.1f}s", flush=True)
        
        if terminated or truncated:
            print("Simulation finished.", flush=True)
            break
            
    env.close()
    print("\nDemo completed.", flush=True)

if __name__ == "__main__":
    demo_run()
