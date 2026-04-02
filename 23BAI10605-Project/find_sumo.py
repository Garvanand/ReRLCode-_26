import os

# User said: C:\Program Files\Eclipse\Sumo
base_path = r"C:\Program Files\Eclipse\Sumo"
sumo_exe = os.path.join(base_path, "bin", "sumo.exe")

print(f"Checking {sumo_exe}")
if os.path.exists(sumo_exe):
    print("Found sumo.exe!")
else:
    print("Not found. Searching...")
    # Try common alternatives
    alternatives = [
        r"C:\Program Files\Sumo\bin\sumo.exe",
        r"C:\Program Files (x86)\Sumo\bin\sumo.exe",
        r"C:\Sumo\bin\sumo.exe",
        r"C:\Program Files\Eclipse\Sumo-1.18.0\bin\sumo.exe", # Example versioned path
    ]
    for alt in alternatives:
        if os.path.exists(alt):
            print(f"Found at: {alt}")
            break
    else:
        print("Still not found.")
