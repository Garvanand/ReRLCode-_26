import os

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    return None

base = r"C:\Program Files (x86)\Eclipse\Sumo\tools"
target = "generateBuildings.py"
result = find_file(target, base)
if result:
    print(f"Found: {result}")
else:
    # Try case insensitive
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower() == target.lower():
                print(f"Found (case-insensitive): {os.path.join(root, f)}")
                break
        else:
            continue
        break
    else:
        print("Not found.")
