import os
import threading
import time
from datetime import datetime

import docker
import matplotlib.pyplot as plt

from .config import PLOT_BASE_PATH


def get_memory_usage(container):
    stats = container.stats(stream=False)
    return stats['memory_stats']['usage']


def get_cpu_usage(container):
    stats = container.stats(stream=False)
    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
    system_cpu_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
    num_cpus = len(stats['cpu_stats']['cpu_usage']['percpu_usage'])
    if system_cpu_delta > 0 and cpu_delta > 0:
        cpu_usage = (cpu_delta / system_cpu_delta) * num_cpus * 100.0
    else:
        cpu_usage = 0.0
    return cpu_usage


class ContainerMonitor(threading.Thread):
    def __init__(self, container_id, interval=.1):
        super().__init__()
        self.client = docker.from_env()
        self.container_id = container_id
        self.interval = interval
        self.running = True
        self.memory_usages = []
        self.cpu_usages = []
        self.timestamps = []

    def run(self):
        container = self.client.containers.get(self.container_id)
        print(f"Starting monitoring for container {self.container_id}.")
        while self.running:
            self.timestamps.append(datetime.now())
            memory_usage = get_memory_usage(container)
            cpu_usage = get_cpu_usage(container)
            self.memory_usages.append(memory_usage)
            self.cpu_usages.append(cpu_usage)
        print(f"Stopped monitoring for container {self.container_id}.")
        self.summarize_stats()

    def stop(self):
        self.running = False

    def summarize_stats(self):
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        # Plot memory usage
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, [mem / (1024 ** 2) for mem in self.memory_usages], label='Memory Usage (MB)')
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (MB)')
        plt.title(f'Memory Usage Over Time for Container {self.container_id}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_BASE_PATH, f"{timestamp}-{self.container_id}-MemoryUsage.png"))
        plt.close()

        # Plot CPU usage
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.cpu_usages, label='CPU Usage (%)', color='orange')
        plt.xlabel('Time')
        plt.ylabel('CPU Usage (%)')
        plt.title(f'CPU Usage Over Time for Container {self.container_id}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_BASE_PATH, f"{timestamp}-{self.container_id}-CPUUsage.png"))
        plt.close()


container_mapper = {
    "CHROMA": ContainerMonitor("chromadb"),
    "MILVUS": ContainerMonitor("milvus-standalone"),
    "PGVECTOR": ContainerMonitor("pgvector"),
    "REDIS": ContainerMonitor("redis-stack")
}
