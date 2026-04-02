import sys
import os

print(f"Current Working Directory: {os.getcwd()}")
print(f"Python Executable: {sys.executable}")
print(f"Python sys.path:")
for p in sys.path:
    print(f"  - {p}")

try:
    import traci
    print("SUCCESS: traci imported")
except ImportError as e:
    print(f"FAILURE: {e}")
