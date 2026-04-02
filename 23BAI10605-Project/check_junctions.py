import os
import sys

# Ensure project root is on path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.sumo_config import setup_sumo, get_sumo_binary
setup_sumo()

import traci

env_path = os.path.join(project_root, 'environment', '2x2_grid.sumocfg')

try:
    sumo_bin = get_sumo_binary(gui=False)
    traci.start([sumo_bin, '-c', env_path, '--no-step-log', 'true'])
    ids = sorted(traci.trafficlight.getIDList())
    print(",".join(ids))
    traci.close()
except:
    try:
        import sumolib
        net = sumolib.net.readNet(os.path.join(project_root, 'environment', '2x2_grid.net.xml'))
        tls_ids = [tls.getID() for tls in net.getTrafficLights()]
        print(",".join(sorted(tls_ids)))
    except:
        pass
