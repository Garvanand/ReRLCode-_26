"""
Centralized SUMO configuration and path detection.
All modules should import from here instead of hardcoding SUMO_HOME.
"""
import os
import sys

# ── Project-wide constants ───────────────────────────────────────────────
STATE_DIM = 37       # 12 queue + 12 wait + 12 occupancy + 1 phase
ACTION_DIM = 2       # 0 = keep phase, 1 = switch phase
GRID_SIZE = 6        # 6x6 grid
DEFAULT_STEP_LENGTH = 5
DEFAULT_YELLOW_DURATION = 3
DEFAULT_MAX_EPISODES = 100
DEFAULT_MAX_STEPS = 1000
UPDATE_TIMESTEP = 2000

# ── SUMO_HOME auto-detection ────────────────────────────────────────────
def find_sumo_home():
    """Locate and set SUMO_HOME, returning the path or None."""
    if 'SUMO_HOME' in os.environ:
        return os.environ['SUMO_HOME']

    # Common install locations on Windows
    candidates = [
        os.path.join(os.environ.get('ProgramFiles', r'C:\Program Files'), 'Eclipse', 'Sumo'),
        os.path.join(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'), 'Eclipse', 'Sumo'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Sumo'),
        r'C:\Sumo',
    ]
    for path in candidates:
        if os.path.isdir(path):
            os.environ['SUMO_HOME'] = path
            return path

    return None


def setup_sumo():
    """
    Find SUMO_HOME, add tools to sys.path, and return the path.
    Raises EnvironmentError if SUMO cannot be found.
    """
    sumo_home = find_sumo_home()
    if sumo_home is None:
        raise EnvironmentError(
            "SUMO_HOME is not set and SUMO could not be auto-detected. "
            "Please install SUMO from https://sumo.dlr.de/docs/Downloads.php "
            "and set the SUMO_HOME environment variable."
        )
    tools_dir = os.path.join(sumo_home, 'tools')
    if tools_dir not in sys.path:
        sys.path.append(tools_dir)

    # Also ensure the project root is on the path so relative imports work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    return sumo_home


def get_sumo_binary(gui=False):
    """Return the absolute path to the sumo / sumo-gui binary."""
    sumo_home = setup_sumo()
    name = 'sumo-gui' if gui else 'sumo'
    binary = os.path.join(sumo_home, 'bin', name)
    # On Windows the .exe extension may or may not be present in the path
    if os.path.isfile(binary) or os.path.isfile(binary + '.exe'):
        return binary
    return name  # fall back to PATH lookup
