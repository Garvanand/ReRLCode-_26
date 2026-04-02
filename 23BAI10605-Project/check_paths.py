import os

paths = [
    r"C:\Program Files\Eclipse\Sumo",
    r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe",
    r"environment\2x2_grid.sumocfg"
]

for p in paths:
    print(f"{p}: {os.path.exists(p)}")
