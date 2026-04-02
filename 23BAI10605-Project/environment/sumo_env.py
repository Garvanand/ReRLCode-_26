import os
import sys
import numpy as np

# Ensure project root is on path for config imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config.sumo_config import setup_sumo, get_sumo_binary
setup_sumo()

import traci
import sumolib
import gymnasium as gym
from gymnasium import spaces

class SumoEnv(gym.Env):
    def __init__(self, sumocfg_path, use_gui=False, step_length=5, yellow_duration=3):
        super(SumoEnv, self).__init__()
        self.sumocfg_path = sumocfg_path
        self.use_gui = use_gui
        self.step_length = step_length
        self.yellow_duration = yellow_duration
        
        sumo_binary = get_sumo_binary(gui=use_gui)
        self.sumo_cmd = [sumo_binary, '-c', sumocfg_path,
                         '--no-step-log', 'true',
                         '--waiting-time-memory', '100',
                         '--no-warnings', 'true']
        if use_gui:
            self.sumo_cmd += ['--start', 'true', '--quit-on-end', 'true']
        
        # Initialize simulation to get junction IDs
        traci.start(self.sumo_cmd)
        self.junction_ids = sorted(traci.trafficlight.getIDList())
        
        # Store phase definitions
        self.phases = {}
        for jid in self.junction_ids:
            logic = traci.trafficlight.getAllProgramLogics(jid)[0]
            self.phases[jid] = logic.phases
            
        traci.close()
        
        # Define Action and Observation Space (Gymnasium standard)
        # Using a dictionary for multi-agent support if needed, or flat for single-junction
        # For simplicity, we'll keep it as a custom environment but with gymnasium interface
        self.state_dim = 37 # 12 lanes * 3 metrics + 1 phase
        self.action_dim = 2 # 0: Keep, 1: Switch
        
        # Placeholder for standard Gym spaces
        self.observation_space = spaces.Dict({
            jid: spaces.Box(low=0, high=np.inf, shape=(37,), dtype=np.float32) 
            for jid in self.junction_ids
        })
        self.action_space = spaces.Dict({
            jid: spaces.Discrete(2) 
            for jid in self.junction_ids
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except:
            pass
        traci.start(self.sumo_cmd)
        return self._get_states(), {}

    def step(self, actions_dict):
        # Handle transitions
        switched = False
        for jid, action in actions_dict.items():
            curr_phase = traci.trafficlight.getPhase(jid)
            if action == 1: # Switch requested
                switched = True
                num_phases = len(self.phases[jid])
                next_phase = (curr_phase + 1) % num_phases
                traci.trafficlight.setPhase(jid, next_phase)
        
        # Run yellow duration if any switch occurred
        for _ in range(self.yellow_duration):
            traci.simulationStep()
            
        # Move to the next green phase if we were in yellow
        for jid, action in actions_dict.items():
            if action == 1:
                curr_phase = traci.trafficlight.getPhase(jid)
                num_phases = len(self.phases[jid])
                next_phase = (curr_phase + 1) % num_phases
                traci.trafficlight.setPhase(jid, next_phase)

        # Run remaining step length
        remaining = self.step_length - self.yellow_duration
        if remaining > 0:
            for _ in range(remaining):
                traci.simulationStep()
        else:
            if not switched:
                for _ in range(self.step_length):
                    traci.simulationStep()
            else:
                traci.simulationStep()
        
        states = self._get_states()
        rewards = self._calculate_rewards()
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = False # Can implement based on time limit if needed
        
        return states, rewards, terminated, truncated, {}

    def _get_states(self):
        states_dict = {}
        for jid in self.junction_ids:
            lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(jid)))
            queues = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
            waits = [traci.lane.getWaitingTime(l) for l in lanes]
            occupancy = [traci.lane.getLastStepOccupancy(l) for l in lanes]
            
            def pad(lst, size):
                return (lst + [0.0]*size)[:size]
            
            state = np.array(pad(queues, 12) + pad(waits, 12) + pad(occupancy, 12) + [float(traci.trafficlight.getPhase(jid))])
            states_dict[jid] = state
        return states_dict

    def _calculate_rewards(self):
        rewards_dict = {}
        for jid in self.junction_ids:
            lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(jid)))
            total_queue = sum([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
            total_wait = sum([traci.lane.getWaitingTime(l) for l in lanes])
            reward = - (0.1 * total_wait + 1.0 * total_queue)
            rewards_dict[jid] = reward
        return rewards_dict

    def close(self):
        try:
            traci.close()
        except:
            pass
