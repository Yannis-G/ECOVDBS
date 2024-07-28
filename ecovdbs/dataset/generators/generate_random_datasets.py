# This file was copied from the ANN Filtered Retrieval Datasets repository.
# Original Author: Qdrant
# Source:
# https://github.com/qdrant/ann-filtering-benchmark-datasets/blob/master/generators/random_data/generate_random_int_datasets.py
# https://github.com/qdrant/ann-filtering-benchmark-datasets/blob/master/generators/random_data/generate_random_keyword_datasets.py
# Modifications: Moved both files in one, Change the path, make it a function, change payload_gen, add top variable
import os
from functools import partial

from ...config import DATA_BASE_PATH
from .generate import DataGenerator, generate_random_dataset


def generate_random_100_keyword_datasets(size=1_000_000, dim=100, name="random_keywords_1m", num_queries=10_000,
                                         top=25):
    generator = DataGenerator(vocab_size=1000)

    # --------------------
    #   KEYWORD PAYLOAD
    # --------------------

    generate_random_dataset(
        generator=generator,
        size=size,
        dim=dim,
        path=os.path.join(DATA_BASE_PATH, name),
        num_queries=num_queries,
        payload_gen=lambda: {
            "a": generator.sample_keyword()
        },
        condition_gen=generator.random_match_keyword,
        top=top
    )


def generate_random_2048_keyword_datasets(size=100_000, dim=2048, name="random_keywords_100k", num_queries=10_000,
                                          top=25):
    generator = DataGenerator(vocab_size=1000)

    # --------------------
    #   KEYWORD PAYLOAD
    # --------------------

    generate_random_dataset(
        generator=generator,
        size=size,
        dim=dim,
        path=os.path.join(DATA_BASE_PATH, name),
        num_queries=num_queries,
        payload_gen=lambda: {
            "a": generator.sample_keyword()
        },
        condition_gen=generator.random_match_keyword,
        top=top
    )


def generate_random_100_int_datasets(size=1_000_000, dim=100, name="random_ints_1m", num_queries=10_000, top=25):
    generator = DataGenerator(vocab_size=1000)

    # --------------------
    #   INT PAYLOAD
    # --------------------

    generate_random_dataset(
        generator=generator,
        size=size,
        dim=dim,
        path=os.path.join(DATA_BASE_PATH, name),
        num_queries=num_queries,
        payload_gen=lambda: {
            "a": generator.random_int(100)
        },
        condition_gen=partial(generator.random_match_int, rng=100),
        top=top
    )


def generate_random_2048_int_datasets(size=100_000, dim=2048, name="random_ints_100k", num_queries=10_000, top=25):
    generator = DataGenerator(vocab_size=1000)

    # --------------------
    #   INT PAYLOAD
    # --------------------

    generate_random_dataset(
        generator=generator,
        size=size,
        dim=dim,
        path=os.path.join(DATA_BASE_PATH, name),
        num_queries=num_queries,
        payload_gen=lambda: {
            "a": generator.random_int(100)
        },
        condition_gen=partial(generator.random_match_int, rng=100),
        top=top
    )
