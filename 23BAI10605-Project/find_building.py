import os

base = r"C:\Program Files (x86)\Eclipse\Sumo\tools"
for root, dirs, files in os.walk(base):
    for f in files:
        if "building" in f.lower():
            print(os.path.join(root, f))
