import os
import sys

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

import traci
import sumolib

sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo.exe')
sumocfg = os.path.abspath("environment/2x2_grid.sumocfg")

print(f"SUMO_HOME: {os.environ['SUMO_HOME']}")
print(f"Binary: {sumo_binary}")
print(f"Config: {sumocfg}")

try:
    traci.start([sumo_binary, "-c", sumocfg])
    print("TraCI started successfully!")
    print(f"Junction IDs: {traci.trafficlight.getIDList()}")
    traci.close()
    print("TraCI closed successfully.")
except Exception as e:
    print(f"Error: {e}")
