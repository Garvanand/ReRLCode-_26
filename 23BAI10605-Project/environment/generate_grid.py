import os
import sys
import subprocess
import random

# Ensure project root is on path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.sumo_config import setup_sumo, GRID_SIZE
sumo_home = setup_sumo()

def generate_network():
    if not sumo_home:
        print("SUMO_HOME not set.")
        return False
    
    netgenerate = os.path.join(sumo_home, 'bin', 'netgenerate')
    if not os.path.isfile(netgenerate) and not os.path.isfile(netgenerate + '.exe'):
        netgenerate = 'netgenerate'

    output_net = os.path.join(os.path.dirname(__file__), '2x2_grid.net.xml')
    
    # Generate a varied grid; perturb flags may not exist in older SUMO versions
    cmd = [
        netgenerate, 
        '--grid', 
        f'--grid.number={GRID_SIZE}', 
        '--grid.length=400', 
        '--no-turnarounds', 
        '--default.lanenumber=3',
        '--grid.attach-length=150',
        f'--output-file={output_net}'
    ]
    # Attempt perturbed layout; fall back silently if not supported
    result = subprocess.run(
        cmd + ['--perturb-x', 'norm(0,40)', '--perturb-y', 'norm(0,40)'],
        capture_output=True
    )
    if result.returncode != 0:
        print("Perturb flags not supported; generating regular grid.")
        subprocess.run(cmd)
    
    netconvert = os.path.join(sumo_home, 'bin', 'netconvert')
    if not os.path.isfile(netconvert) and not os.path.isfile(netconvert + '.exe'):
        netconvert = 'netconvert'
        
    cmd_tls = [
        netconvert,
        '-s', output_net,
        '--tls.guess',
        '--tls.join',
        '--tls.default-type', 'actuated',
        '--crossings.guess',
        '--sidewalks.guess',
        '-o', output_net
    ]
    subprocess.run(cmd_tls)
    
    # Manual Building Generation
    generate_manual_buildings()
    
    return os.path.exists(output_net)

def generate_manual_buildings():
    poly_file = os.path.join(os.path.dirname(__file__), 'buildings.poly.xml')
    with open(poly_file, 'w') as f:
        f.write('<additional>\n')
        for x in range(-500, 3000, 200):
            for y in range(-500, 3000, 200):
                if random.random() < 0.4:
                    w = random.uniform(30, 80)
                    h = random.uniform(30, 80)
                    # Use hex colors or standard SUMO colors
                    color = random.choice(["#808080", "#A52A2A", "#D3D3D3", "#000000"])
                    shape = f"{x},{y} {x+w},{y} {x+w},{y+h} {x},{y+h} {x},{y}"
                    f.write(f'    <poly id="b_{x}_{y}" type="building" color="{color}" fill="true" layer="-1" shape="{shape}"/>\n')
        f.write('</additional>\n')

def generate_routes(density='medium'):
    random_trips = os.path.join(sumo_home, 'tools', 'randomTrips.py')
    net_file = os.path.join(os.path.dirname(__file__), '2x2_grid.net.xml')
    output_rou = os.path.join(os.path.dirname(__file__), '2x2_grid.rou.xml')
    vtypes_file = os.path.join(os.path.dirname(__file__), 'vtypes.add.xml')
    
    periods = {
        'low': 2.0,
        'medium': 0.8,
        'high': 0.3
    }
    p = periods.get(density, 0.8)
    
    cmd = [
        sys.executable,
        random_trips,
        '-n', net_file,
        '-o', output_rou,
        '--period', str(p),
        '--fringe-factor', '15',
        '--trip-attributes', 'departSpeed="max" departLane="best"',
        '--additional-files', vtypes_file,
        '--validate'
    ]
    subprocess.run(cmd)
    
    return os.path.exists(output_rou)

def create_sumocfg():
    cfg_file = os.path.join(os.path.dirname(__file__), '2x2_grid.sumocfg')
    content = """<configuration>
    <input>
        <net-file value="2x2_grid.net.xml"/>
        <route-files value="2x2_grid.rou.xml"/>
        <additional-files value="vtypes.add.xml,buildings.poly.xml"/>
    </input>
    <gui_only>
        <gui-settings-file value="view_settings.xml"/>
    </gui_only>
    <time>
        <begin value="0"/>
        <end value="2000"/>
    </time>
</configuration>"""
    with open(cfg_file, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    if generate_network():
        generate_routes('medium')
        create_sumocfg()
        print("Realistic SUMO files generated successfully.")
    else:
        print("Failed to generate SUMO files.")
