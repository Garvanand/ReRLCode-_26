import os
path = r"C:\Program Files (x86)\Eclipse\Sumo"
print(f"Path: {path}")
print(f"Exists: {os.path.exists(path)}")
if os.path.exists(path):
    print(f"Contents: {os.listdir(path)}")
    bin_path = os.path.join(path, "bin")
    if os.path.exists(bin_path):
        print(f"Bin contents: {os.listdir(bin_path)}")
