import os
try:
    path = r"C:\Program Files (x86)\Eclipse\Sumo"
    print(f"Contents of {path}:")
    print(os.listdir(path))
    tools_path = os.path.join(path, "tools")
    if os.path.exists(tools_path):
        print(f"\nContents of {tools_path}:")
        print(os.listdir(tools_path))
        shapes_path = os.path.join(tools_path, "shapes")
        if os.path.exists(shapes_path):
            print(f"\nContents of {shapes_path}:")
            print(os.listdir(shapes_path))
except Exception as e:
    print(e)
