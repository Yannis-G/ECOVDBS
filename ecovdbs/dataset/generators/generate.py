# This file was copied from the ANN Filtered Retrieval Datasets repository.
# Original Author: Qdrant
# Source: https://github.com/qdrant/ann-filtering-benchmark-datasets/blob/master/generators/generate.py
# Modifications: Removed unused methods, changed check_condition functions to one match, add top variable
#                and docstrings and add modify_payload function.
import json
import os
import random
import string
from typing import List

import numpy as np
import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from ...config import DATA_BASE_PATH


class DataGenerator:
    """
    A class to generate random data for ANN filtering benchmark datasets.
    """

    def __init__(self, vocab_size: int = 1000) -> None:
        """
        Initialize the DataGenerator with a vocabulary of random keywords.

        :param vocab_size: Number of random keywords to generate for the vocabulary.
        """
        self.vocab = [self.random_keyword() for _ in range(vocab_size)]

    @staticmethod
    def random_keyword() -> str:
        """
        Generate a random keyword consisting of five letters.

        :return: A random keyword.
        """
        letters = string.ascii_letters
        return "".join(random.sample(letters, 5))

    def sample_keyword(self) -> str:
        """
        Sample a random keyword from the vocabulary.

        :return: A random keyword from the vocabulary.
        """
        return random.choice(self.vocab)

    @staticmethod
    def random_int(rng=100) -> int:
        """
        Generate a random integer within a range.

        :param rng: The upper limit of the random integer range.
        :return: A random integer.
        """
        return random.randint(0, rng)

    def random_match_keyword(self) -> dict:
        """
        Generate a random match keyword condition.

        :return: A dictionary with a random keyword condition.
        """
        return {
            "value": self.sample_keyword()
        }

    def random_match_int(self, rng=100) -> dict:
        """
        Generate a random match integer condition.

        :param rng: The upper limit of the random integer range.
        :return: A dictionary with a random integer condition.
        """
        return {
            "value": self.random_int(rng)
        }

    @staticmethod
    def random_vectors(size, dim) -> np.ndarray:
        """
        Generate a matrix of random vectors.

        :param size: Number of vectors.
        :param dim: Dimension of each vector.
        :return: A numpy array of random vectors.
        """
        return np.random.rand(size, dim).astype(np.float32)

    @staticmethod
    def check_condition(value, condition) -> bool:
        """
        Check if a value matches a given condition.

        :param value: The value to check.
        :param condition: The condition to match.
        :return: True if the value matches the condition, False otherwise.
        """
        return value == condition['value']

    def search(
            self,
            vectors: np.ndarray,
            payloads: List[dict],
            query: np.ndarray,
            conditions: dict = None,
            top: int = 25) -> tuple:
        """
        Search for the top matching vectors based on the query and conditions.

        :param vectors: Matrix of vectors to search.
        :param payloads: List of payloads associated with the vectors.
        :param query: Query vector.
        :param conditions: Conditions to filter the vectors.
        :param top: Number of top results to return.
        :return: Tuple of lists containing the closest vector IDs and their scores.
        """
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
        num_queries: int,
        dim: int,
        vectors: np.ndarray,
        payloads: List[dict],
        path: str,
        condition_generator,
        top: int = 25,
) -> None:
    """
    Generate query samples and save them to a file.

    :param generator: DataGenerator instance to use for generating samples.
    :param num_queries: Number of query samples to generate.
    :param dim: Dimension of each query vector.
    :param vectors: Matrix of vectors to search.
    :param payloads: List of payloads associated with the vectors.
    :param path: File path to save the generated samples.
    :param condition_generator: Function to generate query conditions.
    :param top: Number of top results to return for each query.
    """
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
        generator: DataGenerator,
        size: int,
        dim: int,
        path: str,
        num_queries: int,
        payload_gen,
        condition_gen,
        top: int = 25
) -> None:
    """
    Generate a random dataset and save it to files.

    :param generator: DataGenerator instance to use for generating the dataset.
    :param size: Number of vectors in the dataset.
    :param dim: Dimension of each vector.
    :param path: Directory path to save the generated dataset.
    :param num_queries: Number of query samples to generate.
    :param payload_gen: Function to generate payloads.
    :param condition_gen: Function to generate query conditions.
    :param top: Number of top results to return for each query.
    """
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


def modify_payload(name: str, item_name: str) -> None:
    """
    This function reads the payloads from a JSON lines file, keeps only the specified item name
    in each payload, and writes the modified payloads back to the file.

    :param name: The name of the dataset.
    :param item_name: The key to be retained in each payload.
    """
    payloads_path = os.path.join(DATA_BASE_PATH, name, "payloads.jsonl")
    payloads = []
    with open(payloads_path) as fd:
        for line in fd:
            data = json.loads(line)
            product_type_name = data.get(item_name, '')
            payloads.append({item_name: product_type_name})

    with open(payloads_path, 'w') as fd:
        for entry in payloads:
            fd.write(json.dumps(entry) + '\n')


def cut_dimensions(dimensions: int, name: str) -> None:
    """
    Cut the dimensions of the vectors in the dataset to the specified number.

    :param name: Name of the dataset directory.
    :param dimensions: Number of dimensions to cut the vectors to.
    """
    vectors_path = os.path.join(DATA_BASE_PATH, name, "vectors.npy")
    vectors = np.load(vectors_path, allow_pickle=False)
    vectors = vectors[:, :dimensions]
    np.save(vectors_path, vectors)
