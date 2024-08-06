from ecovdbs.config import RESULT_BASE_PATH
import tkinter as tk
from tkinter import filedialog
from ecovdbs.results.result import plot_results
from ecovdbs.runner.utility import read_hnsw_runner_result


def main() -> None:
    # Initialize tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog to select multiple files
    file_paths = filedialog.askopenfilenames(
        title="Select result files",
        initialdir=RESULT_BASE_PATH,
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )

    # Process each selected file
    results = []
    for file_path in file_paths:
        result = read_hnsw_runner_result(file_path)
        results.append(result)

    # Plot the results
    plot_results(results)


if __name__ == '__main__':
    main()
