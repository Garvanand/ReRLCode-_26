import subprocess
import os

sumo_home = r"C:\Program Files (x86)\Eclipse\Sumo"
netgenerate = os.path.join(sumo_home, 'bin', 'netgenerate.exe')

result = subprocess.run([netgenerate, '--help'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'perturb' in line or 'grid' in line:
        print(line)
