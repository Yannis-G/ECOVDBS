import matplotlib.pyplot as plt

from ..runner.result_config import HNSWRunnerResult


# Recall/ QPS
# Recall/ Latency


def plot_insert_time(results: list[HNSWRunnerResult]):
    times = [result.insert_result.t_insert_index for result in results]
    labels = [type(res.client).__name__ for res in results]

    fig, ax = plt.subplots()
    ax.bar(labels, times)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Insertion and Index Time')
    for time, label in zip(times, labels):
        ax.annotate(f'{time:.2f}', (label, time))
    plt.show()


def plot_query_performance(results: list[HNSWRunnerResult]):
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
        ax.set_title(f'Query Performance for Mode: {mode.name}')
        ax.legend()
        plt.show()


def plot_index_size(results: list[HNSWRunnerResult]):
    index_sizes = [result.index_size for result in results]
    labels = [type(res.client).__name__ for res in results]

    fig, ax = plt.subplots()
    ax.bar(labels, index_sizes)
    ax.set_ylabel('Size (MB)')
    ax.set_title('Index Size')
    for index_size, label in zip(index_sizes, labels):
        ax.annotate(f'{index_size:.2f}', (label, index_size))
    plt.show()


def plot_disk_size(results: list[HNSWRunnerResult]):
    disk_sizes = [result.disk_size for result in results]
    labels = [type(res.client).__name__ for res in results]

    fig, ax = plt.subplots()
    ax.bar(labels, disk_sizes)
    ax.set_ylabel('Size (MB)')
    ax.set_title('Disk Size')
    for disk_size, label in zip(disk_sizes, labels):
        ax.annotate(f'{disk_size:.2f}', (label, disk_size))
    plt.show()
