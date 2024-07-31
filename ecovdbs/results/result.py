import os
import time

import matplotlib.pyplot as plt

from ..config import PLOT_BASE_PATH
from ..runner.result_config import HNSWRunnerResult


def plot_results(results: list[HNSWRunnerResult], timestamp: str = time.strftime('%Y-%m-%d-%H-%M-%S')) -> None:
    plots: list[tuple[plt.Figure, str]] = [plot_insert_time(results), plot_qps_recall(results),
                                           plot_query_time_recall(results), plot_index_size(results),
                                           plot_disk_size(results)]
    for fig, title in plots:
        fig.savefig(os.path.join(PLOT_BASE_PATH, f"{timestamp}-{title}.png"))
        plt.close(fig)


def plot_insert_time(results: list[HNSWRunnerResult]) -> (plt.Figure, str):
    """
    Plot insertion time for each runner in the results.

    :param results: List of HNSWRunnerResult objects.
    """
    times = [result.insert_result.t_insert_index for result in results]
    labels = [type(res.client).__name__ for res in results]

    fig, ax = plt.subplots()
    ax.bar(labels, times)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Insertion and Index Time')
    for t, label in zip(times, labels):
        ax.annotate(f'{t:.2f}', (label, t))
    return fig, "InsertionIndexTime"


def plot_qps_recall(results: list[HNSWRunnerResult]) -> (plt.Figure, str):
    """
    Plot Queries Per Second (QPS) against Average Recall for each mode.

    :param results: List of HNSWRunnerResult objects.
    """
    # Extract all unique modes from the results
    modes = {mode_result.mode for result in results for mode_result in result.query_result.mode_results}

    # Plot data for each mode separately
    for mode in modes:
        fig, ax = plt.subplots()

        for result in results:
            runner_label = type(result.client).__name__
            for mode_result in result.query_result.mode_results:
                if mode_result.mode == mode:
                    recalls = []
                    qps = []
                    for ef_result in mode_result.ef_results:
                        recalls.append(ef_result.avg_recall)
                        qps.append(ef_result.queries_per_second)
                        ax.annotate(ef_result.ef, (ef_result.avg_recall, ef_result.queries_per_second))
                    ax.plot(recalls, qps, marker='o', label=runner_label)

        ax.set_xlabel('Average Recall')
        ax.set_ylabel('Queries Per Second')
        ax.set_title(f'Queries Per Second/Recall for Mode: {mode.name}')
        ax.legend()
        return fig, "QPSRecall"


def plot_query_time_recall(results: list[HNSWRunnerResult]) -> (plt.Figure, str):
    """
    Plot Average Query Time against Average Recall for each mode.

    :param results: List of HNSWRunnerResult objects.
    """
    # Extract all unique modes from the results
    modes = {mode_result.mode for result in results for mode_result in result.query_result.mode_results}

    # Plot data for each mode separately
    for mode in modes:
        fig, ax = plt.subplots()

        for result in results:
            runner_label = type(result.client).__name__
            for mode_result in result.query_result.mode_results:
                if mode_result.mode == mode:
                    recalls = []
                    avg_query_times = []
                    for ef_result in mode_result.ef_results:
                        recalls.append(ef_result.avg_recall)
                        avg_query_times.append(ef_result.avg_query_time)
                        ax.annotate(ef_result.ef, (ef_result.avg_recall, ef_result.avg_query_time))
                    ax.plot(recalls, avg_query_times, marker='o', label=runner_label)

        ax.set_xlabel('Average Recall')
        ax.set_ylabel('Average Query Time (seconds)')
        ax.set_title(f'Average Query Time (seconds)/Recall for Mode: {mode.name}')
        ax.legend()
        return fig, "QueryTimeRecall"


def plot_index_size(results: list[HNSWRunnerResult]) -> (plt.Figure, str):
    """
    Plot index size for each runner in the results.

    :param results: List of HNSWRunnerResult objects.
    """
    index_sizes = [result.index_size for result in results]
    labels = [type(res.client).__name__ for res in results]

    fig, ax = plt.subplots()
    ax.bar(labels, index_sizes)
    ax.set_ylabel('Size (MB)')
    ax.set_title('Index Size')
    for index_size, label in zip(index_sizes, labels):
        ax.annotate(f'{index_size:.2f}', (label, index_size))
    return fig, "IndexSize"


def plot_disk_size(results: list[HNSWRunnerResult]) -> (plt.Figure, str):
    """
    Plot disk size for each runner in the results.

    :param results: List of HNSWRunnerResult objects.
    """
    disk_sizes = [result.disk_size for result in results]
    labels = [type(res.client).__name__ for res in results]

    fig, ax = plt.subplots()
    ax.bar(labels, disk_sizes)
    ax.set_ylabel('Size (MB)')
    ax.set_title('Disk Size')
    for disk_size, label in zip(disk_sizes, labels):
        ax.annotate(f'{disk_size:.2f}', (label, disk_size))
    return fig, "DiskSize"
