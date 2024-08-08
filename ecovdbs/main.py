import argparse
import logging
from argparse import Namespace

from .dataset.dataset import Dataset
from .dataset.dataset_reader import dataset_mapper
from .docker_stats import container_mapper, ContainerMonitor
from .results.result import plot_results
from .runner.case_config import IndexTime, QueryMode, HNSWCase, HNSWConfig
from .runner.result_config import HNSWRunnerResult
from .runner.runner import HNSWRunner
from .runner.task_config import HNSWTask
from .runner.utility import client_mapper, save_hnsw_runner_result


def print_enum_keys(enum_class, enum_name: str) -> None:
    keys = [e.name.lower() for e in enum_class]
    print(f"Possible {enum_name} values: {', '.join(keys)}")


def print_dict_keys(d: dict, dict_name: str) -> None:
    keys = [key.lower() for key in d.keys()]
    print(f"Possible {dict_name} values: {', '.join(keys)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Instantiate clients and read datasets by name")
    parser.add_argument(
        "--dataset", type=str,
        help="Dataset name (in lowercase). E.g., --dataset sift_small"
    )
    parser.add_argument(
        "--dataset-list", action='store_true',
        help="Print a list of all possible dataset values and exit"
    )
    parser.add_argument(
        "--clients", nargs='+',
        help="List of client names (in lowercase). E.g., --clients chroma milvus"
    )
    parser.add_argument(
        "--clients-list", action='store_true',
        help="Print a list of all possible client values and exit"
    )
    parser.add_argument(
        "--index-time", type=str, default='pre_index',
        help="Index time (in lowercase). Default is pre_index. E.g., --index-time pre_index"
    )
    parser.add_argument(
        "--index-time-list", action='store_true',
        help="Print a list of all possible index-time values and exit"
    )
    parser.add_argument(
        "--query-mode", type=str, default='query',
        help="Query mode (in lowercase). Default is query. E.g., --query-mode query"
    )
    parser.add_argument(
        "--query-modes-list", action='store_true',
        help="Print a list of all possible query-modes values and exit"
    )

    args: Namespace = parser.parse_args()

    # Handle the --dataset-list argument
    if args.dataset_list:
        print_dict_keys(dataset_mapper, "dataset")
        return

    # Handle the --clients-list argument
    if args.clients_list:
        print_dict_keys(client_mapper, "client")
        return

    # Handle the --index-time-list argument
    if args.index_time_list:
        print_enum_keys(IndexTime, "index-time")
        return

    # Handle the --query-modes-list argument
    if args.query_modes_list:
        print_enum_keys(QueryMode, "query-modes")
        return

    # Convert dataset input to uppercase to match the dictionary keys
    dataset_key: str = args.dataset.upper()

    # Convert client inputs to uppercase to match the dictionary keys
    client_keys: list[str] = [name.upper() for name in args.clients]

    # Convert index-time input to uppercase to match the Enum keys
    index_time_key: str = args.index_time.upper()

    # Convert query-mode inputs to uppercase to match the Enum keys
    query_mode_key: str = args.query_mode.upper()

    # Process the dataset
    if dataset_key in dataset_mapper.keys():
        dataset: Dataset = dataset_mapper[dataset_key]()
        print(f"Successfully read dataset: {dataset_key.lower()}")
    else:
        print(f"Error: {dataset_key.lower()} is not a valid dataset name.")
        return

    # Process index-time
    if index_time_key and index_time_key in IndexTime.__members__:
        index_time_value: IndexTime = IndexTime[index_time_key]
        print(f"Index time set to: {index_time_key.lower()}")
    else:
        print(f"Error: {index_time_key.lower()} is not a valid index time.")
        return

    # Process query-modes
    if query_mode_key and query_mode_key in QueryMode.__members__:
        query_mode: QueryMode = QueryMode[query_mode_key]
        print(f"Query mode set to: {query_mode_key.lower()}")
    else:
        print(f"Error: {query_mode_key.lower()} is not a valid query mode.")
        return

    # Process clients
    client_tasks: list[HNSWTask] = []
    container: list[ContainerMonitor] = []
    for client_key in client_keys:
        if client_key in client_mapper.keys():
            client_tasks.append(client_mapper[client_key])
            print(f"Successfully instantiated client: {client_key.lower()}")
        if client_key in container_mapper.keys():
            container.append(container_mapper[client_key])
        else:
            print(f"Error: {client_key.lower()} is not a valid client name.")
            return

    case: HNSWCase = HNSWCase(dataset, HNSWConfig(), index_time_value, query_mode)

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("ecovdbs.runner.runner").setLevel(logging.INFO)

    results: list[HNSWRunnerResult] = []
    for task, monitor in zip(client_tasks, container):
        monitor.start()
        runner: HNSWRunner = HNSWRunner(task(case))
        res: HNSWRunnerResult = runner.run()
        monitor.stop()
        results.append(res)
        save_hnsw_runner_result(res)
    plot_results(results)
