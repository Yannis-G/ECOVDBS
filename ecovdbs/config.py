import os


def find_project_root(current_dir):
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, '.project_root')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return current_dir


BASE_PATH = find_project_root(os.path.abspath(os.path.dirname(__file__)))
RESULT_BASE_PATH = os.path.join(BASE_PATH, "results")
PLOT_BASE_PATH = os.path.join(BASE_PATH, "plots")
DATA_BASE_PATH = os.path.join(BASE_PATH, "data")

if not os.path.exists(RESULT_BASE_PATH):
    os.makedirs(RESULT_BASE_PATH)
if not os.path.exists(PLOT_BASE_PATH):
    os.makedirs(PLOT_BASE_PATH)
if not os.path.exists(DATA_BASE_PATH):
    os.makedirs(DATA_BASE_PATH)
