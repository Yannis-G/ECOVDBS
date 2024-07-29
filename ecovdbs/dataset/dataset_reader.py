import json
import logging
import os
from typing import Callable

import h5py
import numpy as np

from .dataset import Dataset
from .generators.generate_arxiv_queries import modify_tests_and_payload_arxiv
from .generators.generate_hnm_queries import generate_hnm_queries_from_file, modify_filters_and_payload_hnm
from .generators.generate_random_datasets import (generate_random_100_keyword_datasets,
                                                  generate_random_2048_keyword_datasets,
                                                  generate_random_100_int_datasets, generate_random_2048_int_datasets)
from .utility import download, ivecs_read, fvecs_read
from ..client.base_config import MetricType
from ..config import DATA_BASE_PATH

# Constants for dataset file names and download URLs
SIFT_SMALL_128_EUCLIDEAN_FILE = "siftsmall.tar.gz"
SIFT_SMALL_128_EUCLIDEAN_NAME = "siftsmall"
SIFT_SMALL_128_EUCLIDEAN_DOWNLOAD_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
SIFT_128_EUCLIDEAN_FILE = "sift.tar.gz"
SIFT_128_EUCLIDEAN_NAME = "sift"
SIFT_128_EUCLIDEAN_DOWNLOAD_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
GIST_960_EUCLIDEAN_FILE = "gist.tar.gz"
GIST_960_EUCLIDEAN_NAME = "gist"
GIST_960_EUCLIDEAN_DOWNLOAD_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
GLOVE_25_ANGULAR_FILE = "glove-25-angular.hdf5"
GLOVE_25_ANGULAR_DOWNLOAD_URL = "https://ann-benchmarks.com/glove-25-angular.hdf5"
GLOVE_50_ANGULAR_FILE = "glove-50-angular.hdf5"
GLOVE_50_ANGULAR_DOWNLOAD_URL = "https://ann-benchmarks.com/glove-50-angular.hdf5"
GLOVE_100_ANGULAR_FILE = "glove-100-angular.hdf5"
GLOVE_100_ANGULAR_DOWNLOAD_URL = "https://ann-benchmarks.com/glove-100-angular.hdf5"
GLOVE_200_ANGULAR_FILE = "glove-200-angular.hdf5"
GLOVE_200_ANGULAR_DOWNLOAD_URL = "https://ann-benchmarks.com/glove-200-angular.hdf5"
MNIST_784_EUCLIDEAN_FILE = "mnist-784-euclidean.hdf5"
MNIST_784_EUCLIDEAN_DOWNLOAD_URL = "https://ann-benchmarks.com/mnist-784-euclidean.hdf5"
FASHION_MNIST_784_EUCLIDEAN_FILE = "fashion-mnist-784-euclidean.hdf5"
FASHION_MNIST_784_EUCLIDEAN_DOWNLOAD_URL = "https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
DEEP_IMAGE_96_ANGULAR_FILE = "deep-image-96-angular.hdf5"
DEEP_IMAGE_96_ANGULAR_DOWNLOAD_URL = "https://ann-benchmarks.com/deep-image-96-angular.hdf5"

ARXIV_TITLES_384_ANGULAR_KEYWORD_FILE = "arxiv.tar.gz"
ARXIV_TITLES_384_ANGULAR_KEYWORD_NAME = "arxiv"
ARXIV_TITLES_384_ANGULAR_KEYWORD_DOWNLOAD_URL = "https://storage.googleapis.com/ann-filtered-benchmark/datasets/arxiv.tar.gz"
H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_FILE = "hnm.tgz"
H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_NAME = "hnm"
H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_DOWNLOAD_URL = "https://storage.googleapis.com/ann-filtered-benchmark/datasets/hnm.tgz"

RANDOM_100_ANGULAR_KEYWORD_NAME = "random_keywords_1m"
RANDOM_100_ANGULAR_INT_NAME = "random_ints_1m"
RANDOM_2048_ANGULAR_KEYWORD_NAME = "random_keywords_100k"
RANDOM_2048_ANGULAR_INT_NAME = "random_ints_100k"

log = logging.getLogger(__name__)


def download_sift_small() -> None:
    """
    Download and extract the SIFT small dataset.
    """
    _download_tar_file(SIFT_SMALL_128_EUCLIDEAN_FILE, SIFT_SMALL_128_EUCLIDEAN_DOWNLOAD_URL,
                       SIFT_SMALL_128_EUCLIDEAN_NAME, False)


def read_sift_small() -> Dataset:
    """
    Read the SIFT small dataset.
    """
    return _read_tar_file_texmex(download_sift_small, SIFT_SMALL_128_EUCLIDEAN_NAME, 128, MetricType.L2)


def download_sift() -> None:
    """
    Download and extract the SIFT dataset.
    """
    _download_tar_file(SIFT_128_EUCLIDEAN_FILE, SIFT_128_EUCLIDEAN_DOWNLOAD_URL, SIFT_128_EUCLIDEAN_NAME, False)


def read_sift() -> Dataset:
    """
    Read the SIFT dataset.
    """
    return _read_tar_file_texmex(download_sift, SIFT_128_EUCLIDEAN_NAME, 128, MetricType.L2)


def download_gist() -> None:
    """
    Download and extract the GIST dataset.
    """
    _download_tar_file(GIST_960_EUCLIDEAN_FILE, GIST_960_EUCLIDEAN_DOWNLOAD_URL, GIST_960_EUCLIDEAN_NAME, False)


def read_gist() -> Dataset:
    """
    Read the GIST dataset.
    """
    return _read_tar_file_texmex(download_gist, GIST_960_EUCLIDEAN_NAME, 960, MetricType.L2)


def download_glove_25() -> None:
    """
    Download the GloVe 25-dimensional dataset.
    """
    _download_hdf5_file(GLOVE_25_ANGULAR_FILE, GLOVE_25_ANGULAR_DOWNLOAD_URL)


def read_glove_25() -> Dataset:
    """
    Read the GloVe 25-dimensional dataset.
    """
    return _read_hdf5_file_ann_benchmark(download_glove_25, GLOVE_25_ANGULAR_FILE, 25, MetricType.COSINE)


def download_glove_50() -> None:
    """
    Download the GloVe 50-dimensional dataset.
    """
    _download_hdf5_file(GLOVE_50_ANGULAR_FILE, GLOVE_50_ANGULAR_DOWNLOAD_URL)


def read_glove_50() -> Dataset:
    """
    Read the GloVe 50-dimensional dataset.
    """
    return _read_hdf5_file_ann_benchmark(download_glove_50, GLOVE_50_ANGULAR_FILE, 50, MetricType.COSINE)


def download_glove_100() -> None:
    """
    Download the GloVe 100-dimensional dataset.
    """
    _download_hdf5_file(GLOVE_100_ANGULAR_FILE, GLOVE_100_ANGULAR_DOWNLOAD_URL)


def read_glove_100() -> Dataset:
    """
    Read the GloVe 100-dimensional dataset.
    """
    return _read_hdf5_file_ann_benchmark(download_glove_100, GLOVE_100_ANGULAR_FILE, 100, MetricType.COSINE)


def download_glove_200() -> None:
    """
    Download the GloVe 200-dimensional dataset.
    """
    _download_hdf5_file(GLOVE_200_ANGULAR_FILE, GLOVE_200_ANGULAR_DOWNLOAD_URL)


def read_glove_200() -> Dataset:
    """
    Read the GloVe 200-dimensional dataset.
    """
    return _read_hdf5_file_ann_benchmark(download_glove_200, GLOVE_200_ANGULAR_FILE, 200, MetricType.COSINE)


def download_mnist() -> None:
    """
    Download the MNIST dataset.
    """
    _download_hdf5_file(MNIST_784_EUCLIDEAN_FILE, MNIST_784_EUCLIDEAN_DOWNLOAD_URL)


def read_mnist() -> Dataset:
    """
    Read the MNIST dataset.
    """
    return _read_hdf5_file_ann_benchmark(download_mnist, MNIST_784_EUCLIDEAN_FILE, 784, MetricType.L2)


def download_fashion_mnist() -> None:
    """
    Download the Fashion MNIST dataset.
    """
    _download_hdf5_file(FASHION_MNIST_784_EUCLIDEAN_FILE, FASHION_MNIST_784_EUCLIDEAN_DOWNLOAD_URL)


def read_fashion_mnist() -> Dataset:
    """
    Read the Fashion MNIST dataset.
    """
    return _read_hdf5_file_ann_benchmark(download_fashion_mnist, FASHION_MNIST_784_EUCLIDEAN_FILE, 784, MetricType.L2)


def download_deep_image():
    """
    Download the Deep Image dataset.
    """
    _download_hdf5_file(DEEP_IMAGE_96_ANGULAR_FILE, DEEP_IMAGE_96_ANGULAR_DOWNLOAD_URL)


def read_deep_image() -> Dataset:
    """
    Read the Deep Image dataset.
    """
    return _read_hdf5_file_ann_benchmark(download_deep_image, DEEP_IMAGE_96_ANGULAR_FILE, 96, MetricType.COSINE)


def download_arxiv_titles_384_angular() -> None:
    """
    Download the ArXiv titels dataset.
    """
    if not os.path.exists(os.path.join(DATA_BASE_PATH, ARXIV_TITLES_384_ANGULAR_KEYWORD_NAME)):
        _download_tar_file(ARXIV_TITLES_384_ANGULAR_KEYWORD_FILE, ARXIV_TITLES_384_ANGULAR_KEYWORD_DOWNLOAD_URL,
                           ARXIV_TITLES_384_ANGULAR_KEYWORD_NAME)
        modify_tests_and_payload_arxiv(ARXIV_TITLES_384_ANGULAR_KEYWORD_NAME)


def read_arxiv_titles_384_angular() -> Dataset:
    """
    Read the ArXiv titels dataset.
    """
    return _read_filtered_dataset_qdrant(ARXIV_TITLES_384_ANGULAR_KEYWORD_NAME, 384, MetricType.COSINE)


def download_h_and_m_clothes_2048_angular() -> None:
    """
    Download and process the H&M clothes dataset.
    """
    if not os.path.exists(os.path.join(DATA_BASE_PATH, H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_NAME)):
        _download_tar_file(H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_FILE, H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_DOWNLOAD_URL,
                           H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_NAME)
        log.info(
            "Modifying filters and payloads for H&M clothes dataset and generating HNM queries for H&M clothes dataset")
        modify_filters_and_payload_hnm(H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_NAME)
        generate_hnm_queries_from_file(name=H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_NAME)


def read_h_and_m_clothes_2048_angular() -> Dataset:
    """
    Read the H&M clothes dataset.
    """
    return _read_filtered_dataset_qdrant(H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_NAME, 2048, MetricType.COSINE)


def download_random_100_angular_keyword() -> None:
    """
    Generate random 100-dimensional keyword datasets.
    """
    log.info("Generating random 100-dimensional keyword datasets")
    if not os.path.exists(os.path.join(DATA_BASE_PATH, RANDOM_100_ANGULAR_KEYWORD_NAME)):
        generate_random_100_keyword_datasets(name=RANDOM_100_ANGULAR_KEYWORD_NAME)


def read_random_100_angular_keyword() -> Dataset:
    """
    Read random 100-dimensional keyword datasets.
    """
    return _read_filtered_dataset_qdrant(RANDOM_100_ANGULAR_KEYWORD_NAME, 100, MetricType.COSINE)


def download_random_100_angular_int() -> None:
    """
    Generate random 100-dimensional integer datasets.
    """
    log.info("Generating random 100-dimensional integer datasets")
    if not os.path.exists(os.path.join(DATA_BASE_PATH, RANDOM_100_ANGULAR_INT_NAME)):
        generate_random_100_int_datasets(name=RANDOM_100_ANGULAR_INT_NAME)


def read_random_100_angular_int() -> Dataset:
    """
    Read random 100-dimensional integer datasets.
    """
    return _read_filtered_dataset_qdrant(RANDOM_100_ANGULAR_INT_NAME, 100, MetricType.COSINE)


def download_random_2048_angular_keyword() -> None:
    """
    Generate random 2048-dimensional keyword datasets.
    """
    log.info("Generating random 2048-dimensional keyword datasets")
    if not os.path.exists(os.path.join(DATA_BASE_PATH, RANDOM_2048_ANGULAR_KEYWORD_NAME)):
        generate_random_2048_keyword_datasets(name=RANDOM_2048_ANGULAR_KEYWORD_NAME)


def read_random_2048_angular_keyword() -> Dataset:
    """
    Read random 2048-dimensional keyword datasets.
    """
    return _read_filtered_dataset_qdrant(RANDOM_2048_ANGULAR_KEYWORD_NAME, 2048, MetricType.COSINE)


def download_random_2048_angular_int() -> None:
    """
    Generate random 2048-dimensional integer datasets.
    """
    log.info("Generating random 2048-dimensional integer datasets")
    if not os.path.exists(os.path.join(DATA_BASE_PATH, RANDOM_2048_ANGULAR_INT_NAME)):
        generate_random_2048_int_datasets(name=RANDOM_2048_ANGULAR_INT_NAME)


def read_random_2048_angular_int() -> Dataset:
    """
    Read random 2048-dimensional integer datasets.
    """
    return _read_filtered_dataset_qdrant(RANDOM_2048_ANGULAR_INT_NAME, 2048, MetricType.COSINE)


def _download_tar_file(file_name: str, url: str, name: str, create_dir: bool = True):
    """
    Download and extract a tar file.
    """
    log.info(f"Downloading {name} dataset")
    if not os.path.exists(os.path.join(DATA_BASE_PATH, name)):
        download(url, os.path.join(DATA_BASE_PATH, file_name))
        if create_dir:
            dest_path = os.path.join(DATA_BASE_PATH, name)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            os.system(f"tar -xvf {os.path.join(DATA_BASE_PATH, file_name)} -C {dest_path}")
        else:
            os.system(f"tar -xvf {os.path.join(DATA_BASE_PATH, file_name)} -C {DATA_BASE_PATH}")
        os.remove(os.path.join(DATA_BASE_PATH, file_name))


def _read_tar_file_texmex(download_func: Callable[[], None], name: str, dimension: int,
                          metric_type: MetricType) -> Dataset:
    """
    Read a dataset from a tar file in the format of http://corpus-texmex.irisa.fr/.
    """
    download_func()
    return Dataset(dimension, metric_type,
                   fvecs_read(os.path.join(DATA_BASE_PATH, f"{name}/{name}_base.fvecs")),
                   fvecs_read(os.path.join(DATA_BASE_PATH, f"{name}/{name}_query.fvecs")),
                   ivecs_read(os.path.join(DATA_BASE_PATH, f"{name}/{name}_groundtruth.ivecs")))


def _download_hdf5_file(file_name: str, url: str):
    """
    Download an HDF5 file.
    """
    log.info(f"Downloading {file_name} dataset")
    if not os.path.exists(os.path.join(DATA_BASE_PATH, file_name)):
        download(url, os.path.join(DATA_BASE_PATH, file_name))


def _read_hdf5_file_ann_benchmark(download_func: Callable[[], None], file_name: str, dimension: int,
                                  metric_type: MetricType) -> Dataset:
    """
    Read a dataset from an HDF5 file in the format of https://github.com/erikbern/ann-benchmarks.
    """
    download_func()
    with h5py.File(os.path.join(DATA_BASE_PATH, file_name), 'r') as f:
        return Dataset(dimension, metric_type,
                       f["train"][()].tolist(),
                       f["test"][()].tolist(),
                       f["neighbors"][()].tolist())


def _read_filtered_dataset_qdrant(name: str, dimension: int, metric_type: MetricType) -> Dataset:
    """
    Read a filtered dataset in the format of https://github.com/qdrant/ann-filtering-benchmark-datasets.
    """
    vectors_path = os.path.join(DATA_BASE_PATH, name, "vectors.npy")
    vectors = np.load(vectors_path, allow_pickle=False).tolist()

    payloads_path = os.path.join(DATA_BASE_PATH, name, "payloads.jsonl")
    payloads = []
    with open(payloads_path) as fd:
        for line in fd:
            data = json.loads(line)
            payloads.append(list(data.values())[0])

    test_path = os.path.join(DATA_BASE_PATH, name, "tests.jsonl")
    print(test_path)
    queries = []
    closest_ids = []
    filters = []
    with open(test_path) as fd:
        for line in fd:
            data = json.loads(line)
            queries.append(data['query'])
            closest_ids.append(data['closest_ids'])
            conditions = data['conditions']
            field = list(conditions.keys())[0]
            filters.append(conditions[field]['value'])

    return Dataset(dimension, metric_type, vectors, queries, closest_ids, payloads, filters)


dataset_mapper = {
    "SIFT_SMALL": read_sift_small,
    "SIFT": read_sift,
    "GIST": read_gist,
    "GLOVE_25": read_glove_25,
    "GLOVE_50": read_glove_50,
    "GLOVE_100": read_glove_100,
    "GLOVE_200": read_glove_200,
    "MNIST": read_mnist,
    "FASHION_MNIST": read_fashion_mnist,
    "DEEP_IMAGE": read_deep_image,
    "ARXIV_TITLES": read_arxiv_titles_384_angular,
    "H_AND_M_CLOTHES": read_h_and_m_clothes_2048_angular,
    "RANDOM_100_KEYWORD": read_random_100_angular_keyword,
    "RANDOM_100_INT": read_random_100_angular_int,
    "RANDOM_2048_KEYWORD": read_random_2048_angular_keyword,
    "RANDOM_2048_INT": read_random_2048_angular_int
}
