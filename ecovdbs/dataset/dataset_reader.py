import os
import time
from typing import Callable

import h5py
from .utility import download, ivecs_read, fvecs_read
from .dataset import Dataset
from ..client.base_config import MetricType
from ..config import DATA_BASE_PATH

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
RANDOM_100_ANGULAR_KEYWORD_FILE = "random_keywords_1m.tgz"
RANDOM_100_ANGULAR_KEYWORD_NAME = "random_keywords_1m"
RANDOM_100_ANGULAR_KEYWORD_DOWNLOAD_URL = "https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_keywords_1m.tgz"
RANDOM_100_ANGULAR_INT_FILE = "random_ints_1m.tgz"
RANDOM_100_ANGULAR_INT_NAME = "random_ints_1m"
RANDOM_100_ANGULAR_INT_DOWNLOAD_URL = "https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_ints_1m.tgz"
RANDOM_2048_ANGULAR_KEYWORD_FILE = "random_keywords_100k.tgz"
RANDOM_2048_ANGULAR_KEYWORD_NAME = "random_keywords_100k"
RANDOM_2048_ANGULAR_KEYWORD_DOWNLOAD_URL = "https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_keywords_100k.tgz"
RANDOM_2048_ANGULAR_INT_FILE = "random_ints_100k.tgz"
RANDOM_2048_ANGULAR_INT_NAME = "random_ints_100k"
RANDOM_2048_ANGULAR_INT_DOWNLOAD_URL = "https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_ints_100k.tgz"


def download_sift_small() -> None:
    _download_tar_file(SIFT_SMALL_128_EUCLIDEAN_FILE, SIFT_SMALL_128_EUCLIDEAN_DOWNLOAD_URL,
                       SIFT_SMALL_128_EUCLIDEAN_NAME, False)


def read_sift_small() -> Dataset:
    return _read_tar_file_texmex(download_sift_small, SIFT_SMALL_128_EUCLIDEAN_NAME, 128, MetricType.L2)


def download_sift() -> None:
    _download_tar_file(SIFT_128_EUCLIDEAN_FILE, SIFT_128_EUCLIDEAN_DOWNLOAD_URL, SIFT_128_EUCLIDEAN_NAME, False)


def read_sift() -> Dataset:
    return _read_tar_file_texmex(download_sift, SIFT_128_EUCLIDEAN_NAME, 128, MetricType.L2)


def download_gist() -> None:
    _download_tar_file(GIST_960_EUCLIDEAN_FILE, GIST_960_EUCLIDEAN_DOWNLOAD_URL, GIST_960_EUCLIDEAN_NAME, False)


def read_gist() -> Dataset:
    return _read_tar_file_texmex(download_gist, GIST_960_EUCLIDEAN_NAME, 960, MetricType.L2)


def download_glove_25() -> None:
    _download_hdf5_file(GLOVE_25_ANGULAR_FILE, GLOVE_25_ANGULAR_DOWNLOAD_URL)


def read_glove_25() -> Dataset:
    return _read_hdf5_file_ann_benchmark(download_glove_25, GLOVE_25_ANGULAR_FILE, 25, MetricType.COSINE)


def download_glove_50() -> None:
    _download_hdf5_file(GLOVE_50_ANGULAR_FILE, GLOVE_50_ANGULAR_DOWNLOAD_URL)


def read_glove_50() -> Dataset:
    return _read_hdf5_file_ann_benchmark(download_glove_50, GLOVE_50_ANGULAR_FILE, 50, MetricType.COSINE)


def download_glove_100() -> None:
    _download_hdf5_file(GLOVE_100_ANGULAR_FILE, GLOVE_100_ANGULAR_DOWNLOAD_URL)


def read_glove_100() -> Dataset:
    return _read_hdf5_file_ann_benchmark(download_glove_100, GLOVE_100_ANGULAR_FILE, 100, MetricType.COSINE)


def download_glove_200() -> None:
    _download_hdf5_file(GLOVE_200_ANGULAR_FILE, GLOVE_200_ANGULAR_DOWNLOAD_URL)


def read_glove_200() -> Dataset:
    return _read_hdf5_file_ann_benchmark(download_glove_200, GLOVE_200_ANGULAR_FILE, 200, MetricType.COSINE)


def download_mnist() -> None:
    _download_hdf5_file(MNIST_784_EUCLIDEAN_FILE, MNIST_784_EUCLIDEAN_DOWNLOAD_URL)


def read_mnist() -> Dataset:
    return _read_hdf5_file_ann_benchmark(download_mnist, MNIST_784_EUCLIDEAN_FILE, 784, MetricType.L2)


def download_fashion_mnist() -> None:
    _download_hdf5_file(FASHION_MNIST_784_EUCLIDEAN_FILE, FASHION_MNIST_784_EUCLIDEAN_DOWNLOAD_URL)


def read_fashion_mnist() -> Dataset:
    return _read_hdf5_file_ann_benchmark(download_fashion_mnist, FASHION_MNIST_784_EUCLIDEAN_FILE, 784, MetricType.L2)


def download_deep_image():
    _download_hdf5_file(DEEP_IMAGE_96_ANGULAR_FILE, DEEP_IMAGE_96_ANGULAR_DOWNLOAD_URL)


def read_deep_image() -> Dataset:
    return _read_hdf5_file_ann_benchmark(download_deep_image, DEEP_IMAGE_96_ANGULAR_FILE, 96, MetricType.COSINE)


def download_arxiv_titles_384_angular() -> None:
    _download_tar_file(ARXIV_TITLES_384_ANGULAR_KEYWORD_FILE, ARXIV_TITLES_384_ANGULAR_KEYWORD_DOWNLOAD_URL,
                       ARXIV_TITLES_384_ANGULAR_KEYWORD_NAME)


# TODO read_arxiv_titles_384_angular

def download_h_and_m_clothes_2048_angular() -> None:
    _download_tar_file(H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_FILE, H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_DOWNLOAD_URL,
                       H_AND_M_CLOTHES_2048_ANGULAR_KEYWORD_NAME)


# TODO read_h_and_m_clothes_2048_angular

def download_random_100_angular_keyword() -> None:
    _download_tar_file(RANDOM_100_ANGULAR_KEYWORD_FILE, RANDOM_100_ANGULAR_KEYWORD_DOWNLOAD_URL,
                       RANDOM_100_ANGULAR_KEYWORD_NAME)


# TODO read_random_100_angular_keyword

def download_random_100_angular_int() -> None:
    _download_tar_file(RANDOM_100_ANGULAR_INT_FILE, RANDOM_100_ANGULAR_INT_DOWNLOAD_URL, RANDOM_100_ANGULAR_INT_NAME)


# TODO read_random_100_angular_int

def download_random_2048_angular_keyword() -> None:
    _download_tar_file(RANDOM_2048_ANGULAR_KEYWORD_FILE, RANDOM_2048_ANGULAR_KEYWORD_DOWNLOAD_URL,
                       RANDOM_2048_ANGULAR_KEYWORD_NAME)


# TODO read_random_2048_angular


def download_random_2048_angular_int() -> None:
    _download_tar_file(RANDOM_2048_ANGULAR_INT_FILE, RANDOM_2048_ANGULAR_INT_DOWNLOAD_URL, RANDOM_2048_ANGULAR_INT_NAME)


# TODO read_random_2048_angular_int

def _download_tar_file(file_name: str, url: str, name: str, create_dir: bool = True):
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
    download_func()
    return Dataset(dimension, metric_type,
                   fvecs_read(os.path.join(DATA_BASE_PATH, f"{name}/{name}_base.fvecs")),
                   fvecs_read(os.path.join(DATA_BASE_PATH, f"{name}/{name}_query.fvecs")),
                   ivecs_read(os.path.join(DATA_BASE_PATH, f"{name}/{name}_groundtruth.ivecs")))


def _download_hdf5_file(file_name: str, url: str):
    if not os.path.exists(os.path.join(DATA_BASE_PATH, file_name)):
        download(url, os.path.join(DATA_BASE_PATH, file_name))


def _read_hdf5_file_ann_benchmark(download_func: Callable[[], None], file_name: str, dimension: int,
                                  metric_type: MetricType) -> Dataset:
    download_func()
    with h5py.File(os.path.join(DATA_BASE_PATH, file_name), 'r') as f:
        return Dataset(dimension, metric_type,
                       f["train"][()].tolist(),
                       f["test"][()].tolist(),
                       f["neighbors"][()].tolist())
