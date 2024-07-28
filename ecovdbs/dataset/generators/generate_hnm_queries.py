# This file was copied from the ANN Filtered Retrieval Datasets repository.
# Original Author: Qdrant
# Source:
# https://github.com/qdrant/ann-filtering-benchmark-datasets/blob/master/generators/clothes_images/generate_hnm_queries.py
# Modifications: Changed the path, make it a function, change query to one match, add top variable and add
#                modify_filters_and_payload function
import json
import os
import random
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from ...config import DATA_BASE_PATH
from .generate import DataGenerator


def generate_query(filters: Dict[str, list]):
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
        path,
        top=25
):
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


def convert_filters(filters):
    res = {}
    for field in filters:
        res[field['name']] = field['values']
    return res


def generate_hnm_queries_from_file(name="hnm", num_queries=10_000, top=25):
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


def modify_filters_and_payload(name="hnm"):
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
