import argparse

from .runner.utility import client_mapper
from .runner.case_config import IndexTime, QueryMode, HNSWCase, HNSWConfig

from .dataset.dataset_reader import dataset_mapper


def print_enum_keys(enum_class, enum_name):
    keys = [e.name.lower() for e in enum_class]
    print(f"Possible {enum_name} values: {', '.join(keys)}")


def print_dict_keys(d, dict_name):
    keys = [key.lower() for key in d.keys()]
    print(f"Possible {dict_name} values: {', '.join(keys)}")


def main():
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
        "--clients", nargs='+', required=True,
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
        "--query-mode", nargs='+', default=['query'],
        help="Query mode (in lowercase). Default is query. E.g., --query-mode query"
    )
    parser.add_argument(
        "--query-modes-list", action='store_true',
        help="Print a list of all possible query-modes values and exit"
    )

    args = parser.parse_args()

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
    dataset_key = args.dataset.upper()

    # Convert client inputs to uppercase to match the dictionary keys
    client_keys = [name.upper() for name in args.clients]

    # Convert index-time input to uppercase to match the Enum keys
    index_time_key = args.index_time.upper()

    # Convert query-mode inputs to uppercase to match the Enum keys
    query_mode_key = args.query_mode.upper()

    dataset, index_time_value, query_mode, clients = None, None, None, []

    # Process the dataset
    if dataset_key in dataset_mapper.keys():
        dataset = dataset_mapper[dataset_key]()
        print(f"Successfully read dataset: {dataset_key.lower()}")
    else:
        print(f"Error: {dataset_key.lower()} is not a valid dataset name.")
        return

    # Process index-time
    if index_time_key:
        if index_time_key in IndexTime.__members__:
            index_time_value = IndexTime[index_time_key]
            print(f"Index time set to: {index_time_key.lower()}")
        else:
            print(f"Error: {index_time_key.lower()} is not a valid index time.")
            return

    # Process query-modes
    if query_mode_key:
        if query_mode_key in QueryMode.__members__:
            query_mode = QueryMode[query_mode_key]
            print(f"Query mode set to: {query_mode_key.lower()}")
        else:
            print(f"Error: {query_mode_key.lower()} is not a valid query mode.")
            return

    # Process clients
    for client_key in client_keys:
        if client_key in client_mapper.keys():
            clients.append(client_mapper[client_key])
            print(f"Successfully instantiated client: {client_key.lower()}")
        else:
            print(f"Error: {client_key.lower()} is not a valid client name.")
            return

    case = HNSWCase(dataset, HNSWConfig(), index_time_value, query_mode)

    # TODO: start task for each client with the case
