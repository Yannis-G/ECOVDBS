# This file was copied from the ANN Filtered Retrieval Datasets repository.
# Original Author: Qdrant
# Source: https://github.com/qdrant/ann-filtering-benchmark-datasets/blob/master/generators/generate.py
# Modifications: Removed unused methods, changed check_condition functions to one match and add top variable
import json
import os
import random
import string
from typing import List

import numpy as np
import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class DataGenerator:

    def __init__(self, vocab_size=1000):
        self.vocab = [self.random_keyword() for _ in range(vocab_size)]

    @staticmethod
    def random_keyword():
        letters = string.ascii_letters
        return "".join(random.sample(letters, 5))

    def sample_keyword(self):
        return random.choice(self.vocab)

    @staticmethod
    def random_int(rng=100):
        return random.randint(0, rng)

    def random_match_keyword(self):
        return {
            "value": self.sample_keyword()
        }

    def random_match_int(self, rng=100):
        return {
            "value": self.random_int(rng)
        }

    @staticmethod
    def random_vectors(size, dim):
        return np.random.rand(size, dim).astype(np.float32)

    @staticmethod
    def check_condition(value, condition):
        return value == condition['value']

    def search(
            self,
            vectors: np.ndarray,
            payloads: List[dict],
            query: np.ndarray,
            conditions: dict = None,
            top=25):
        field = list(conditions.keys())[0]
        mask = np.array(
            [self.check_condition(value=payload[field], condition=conditions[field]) for payload in payloads])

        # Select only matched by payload vectors
        filtered_vectors = vectors[mask]
        # List of original ids
        raw_ids = np.arange(0, len(vectors))
        # List of ids, filtered by payload
        filtered_ids = raw_ids[mask]
        if len(filtered_vectors) == 0:
            return [], []
        # Scores among filtered vectors
        scores = cosine_similarity([query], filtered_vectors)[0]
        # Ids in filtered matrix
        top_scores_ids = np.argsort(scores)[-top:][::-1]
        top_scores = scores[top_scores_ids]
        # Original ids before filtering
        original_ids = filtered_ids[top_scores_ids]
        return list(map(int, original_ids)), list(map(float, top_scores))


def generate_samples(
        generator: DataGenerator,
        num_queries,
        dim,
        vectors,
        payloads,
        path,
        condition_generator,
        top=25,
):
    with open(path, "w") as out:
        for _ in tqdm.tqdm(range(num_queries)):
            query = generator.random_vectors(1, dim=dim)[0]
            conditions = {
                'a': condition_generator()
            }

            closest_ids, best_scores = generator.search(
                vectors=vectors,
                payloads=payloads,
                query=query,
                conditions=conditions,
                top=top,
            )

            out.write(json.dumps(
                {
                    "query": query.tolist(),
                    "conditions": conditions,
                    "closest_ids": closest_ids,
                    "closest_scores": best_scores
                }
            ))

            out.write("\n")


def generate_random_dataset(
        generator,
        size,
        dim,
        path,
        num_queries,
        payload_gen,
        condition_gen,
        top=25
):
    os.makedirs(path, exist_ok=True)
    vectors = generator.random_vectors(size, dim)

    np.save(os.path.join(path, "vectors.npy"), vectors, allow_pickle=False)

    payloads = [payload_gen() for _ in range(size)]

    with open(os.path.join(path, "payloads.jsonl"), "w") as out:
        for payload in payloads:
            out.write(json.dumps(payload))
            out.write("\n")

    generate_samples(
        generator=generator,
        num_queries=num_queries,
        dim=dim,
        vectors=vectors,
        payloads=payloads,
        path=os.path.join(path, "tests.jsonl"),
        condition_generator=condition_gen,
        top=top
    )
