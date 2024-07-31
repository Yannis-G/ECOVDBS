import os
from matplotlib import use as mpl_use


def find_project_root(current_dir: str) -> str:
    """
    Recursively find the root directory of the project by looking for a specific marker file.

    :param current_dir: The current directory to start the search from.
    :return: The root directory of the project.
    """
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, '.project_root')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return current_dir


# Set Matplotlib to use the Agg backend
mpl_use('Agg')

# Determine the base path of the project by finding the project root
BASE_PATH = find_project_root(os.path.abspath(os.path.dirname(__file__)))

# Define paths for results, plots, and data directories
RESULT_BASE_PATH = os.path.join(BASE_PATH, "results")
PLOT_BASE_PATH = os.path.join(BASE_PATH, "plots")
DATA_BASE_PATH = os.path.join(BASE_PATH, "data")

# Create the directories if they do not already exist
os.makedirs(RESULT_BASE_PATH, exist_ok=True)
os.makedirs(PLOT_BASE_PATH, exist_ok=True)
os.makedirs(DATA_BASE_PATH, exist_ok=True)
