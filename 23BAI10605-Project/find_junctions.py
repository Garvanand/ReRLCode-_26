with open('environment/2x2_grid.net.xml', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if '<junction' in line:
            print(f"Line {i+1}: {line.strip()}")
        if 'type="traffic_light"' in line:
            print(f"Line {i+1}: {line.strip()}")
