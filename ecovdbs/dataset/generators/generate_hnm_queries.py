# This file was copied from the ANN Filtered Retrieval Datasets repository.
# Original Author: Qdrant
# Source:
# https://github.com/qdrant/ann-filtering-benchmark-datasets/blob/master/generators/clothes_images/generate_hnm_queries.py
# Modifications: Changed the path, make it a function, change query to one match, add top variable, add
#                modify_filters_and_payload function and add docstrings.
import json
import os
import random
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from ...config import DATA_BASE_PATH
from .generate import DataGenerator


def generate_query(filters: Dict[str, list]) -> Dict[str, Dict[str, str]]:
    """
    Generate a query with a random filter condition.

    :param filters: Dictionary containing possible filter values for each field.
    :return: A dictionary representing the query filter.
    """
    fields = list(filters.keys())
    field = random.choice(fields)
    value = random.choice(filters[field])
    return {
        field: {
            "value": value
        }
    }


def generate_hnm_queries(
        vectors: np.ndarray,
        payloads: List[dict],
        filters: Dict[str, list],
        num_queries: int,
        path: str,
        top: int = 25
) -> None:
    """
    Generate hard-negative mining (HNM) queries and save them to a file.

    :param vectors: Array of vectors representing the dataset.
    :param payloads: List of payload dictionaries associated with each vector.
    :param filters: Dictionary containing possible filter values for each field.
    :param num_queries: Number of queries to generate.
    :param path: Path to save the generated queries.
    :param top: Number of top results to return for each query.
    """
    generator = DataGenerator()
    with open(path, "w") as out:
        for _ in tqdm(range(num_queries)):
            ref_id = random.randint(0, len(vectors))
            query_vector = vectors[ref_id]
            query_filter = generate_query(filters=filters)

            closest_ids, best_scores = generator.search(
                vectors=vectors,
                payloads=payloads,
                query=query_vector,
                conditions=query_filter,
                top=top
            )

            out.write(json.dumps(
                {
                    "query": query_vector.tolist(),
                    "conditions": query_filter,
                    "closest_ids": closest_ids,
                    "closest_scores": best_scores
                }
            ))

            out.write("\n")


def convert_filters(filters: List[dict]) -> Dict[str, List[str]]:
    """
    Convert a list of filter dictionaries to a dictionary with filter names as keys and filter values as lists.

    :param filters: List of filter dictionaries.
    :return: Dictionary with filter names as keys and filter values as lists.
    """
    res = {}
    for field in filters:
        res[field['name']] = field['values']
    return res


def generate_hnm_queries_from_file(name: str = "hnm", num_queries: int = 10_000, top: int = 25) -> None:
    """
    Generate HNM queries from pre-saved vectors, payloads, and filters, and save them to a file.

    :param name: Name of the dataset directory.
    :param num_queries: Number of queries to generate.
    :param top: Number of top results to return for each query.
    """
    vectors_path = os.path.join(DATA_BASE_PATH, name, "vectors.npy")
    vectors = np.load(vectors_path, allow_pickle=False)

    payloads_path = os.path.join(DATA_BASE_PATH, name, "payloads.jsonl")
    payloads = []
    with open(payloads_path) as fd:
        for line in fd:
            payloads.append(json.loads(line))

    filters_path = os.path.join(DATA_BASE_PATH, name, "filters.json")
    filters = convert_filters(json.load(open(filters_path)))

    generate_hnm_queries(
        vectors=vectors,
        payloads=payloads,
        filters=filters,
        num_queries=num_queries,
        path=os.path.join(DATA_BASE_PATH, name, "tests.jsonl"),
        top=top
    )


def modify_filters_and_payload(name: str = "hnm") -> None:
    """
    Modify filters and payloads for the dataset to only include 'product_type_name'.

    :param name: Name of the dataset directory.
    """
    filters_path = os.path.join(DATA_BASE_PATH, name, "filters.json")
    filters = json.load(open(filters_path))
    filters = [item for item in filters if item['name'] == 'product_type_name']
    json.dump(filters, open(filters_path, 'w'))

    payloads_path = os.path.join(DATA_BASE_PATH, name, "payloads.jsonl")
    payloads = []
    with open(payloads_path) as fd:
        for line in fd:
            data = json.loads(line)
            product_type_name = data.get('product_type_name', '')
            payloads.append(product_type_name + '\n')

    with open(payloads_path, 'w') as fd:
        fd.writelines(payloads)
