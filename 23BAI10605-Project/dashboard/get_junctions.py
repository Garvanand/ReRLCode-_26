import os
import sys

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.sumo_config import setup_sumo, get_sumo_binary
setup_sumo()

import traci

def get_junction_ids():
    env_path = os.path.join(project_root, 'environment', '2x2_grid.sumocfg')
    sumo_bin = get_sumo_binary(gui=False)
    cmd = [sumo_bin, '-c', env_path, '--no-step-log', 'true']
    try:
        traci.start(cmd)
        junction_ids = traci.trafficlight.getIDList()
        traci.close()
        return junction_ids
    except:
        return []

if __name__ == "__main__":
    ids = get_junction_ids()
    print(",".join(ids))
