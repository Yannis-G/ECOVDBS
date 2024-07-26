import os
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


def download_sift_small() -> None:
    _download_tar_file(SIFT_SMALL_128_EUCLIDEAN_FILE, SIFT_SMALL_128_EUCLIDEAN_DOWNLOAD_URL)


def read_sift_small() -> Dataset:
    return _read_tar_file(download_sift_small, SIFT_SMALL_128_EUCLIDEAN_NAME, 128, MetricType.L2)


def download_sift() -> None:
    _download_tar_file(SIFT_128_EUCLIDEAN_FILE, SIFT_128_EUCLIDEAN_DOWNLOAD_URL)


def read_sift() -> Dataset:
    return _read_tar_file(download_sift, SIFT_128_EUCLIDEAN_NAME, 128, MetricType.L2)


def download_gist() -> None:
    _download_tar_file(GIST_960_EUCLIDEAN_FILE, GIST_960_EUCLIDEAN_DOWNLOAD_URL)


def read_gist() -> Dataset:
    return _read_tar_file(download_gist, GIST_960_EUCLIDEAN_NAME, 960, MetricType.L2)


def download_glove_25() -> None:
    _download_hdf5_file(GLOVE_25_ANGULAR_FILE, GLOVE_25_ANGULAR_DOWNLOAD_URL)


def read_glove_25() -> Dataset:
    return _read_hdf5_file(download_glove_25, GLOVE_25_ANGULAR_FILE, 25, MetricType.COSINE)


def download_glove_50() -> None:
    _download_hdf5_file(GLOVE_50_ANGULAR_FILE, GLOVE_50_ANGULAR_DOWNLOAD_URL)


def read_glove_50() -> Dataset:
    return _read_hdf5_file(download_glove_50, GLOVE_50_ANGULAR_FILE, 50, MetricType.COSINE)


def download_glove_100() -> None:
    _download_hdf5_file(GLOVE_100_ANGULAR_FILE, GLOVE_100_ANGULAR_DOWNLOAD_URL)


def read_glove_100() -> Dataset:
    return _read_hdf5_file(download_glove_100, GLOVE_100_ANGULAR_FILE, 100, MetricType.COSINE)


def download_glove_200() -> None:
    _download_hdf5_file(GLOVE_200_ANGULAR_FILE, GLOVE_200_ANGULAR_DOWNLOAD_URL)


def read_glove_200() -> Dataset:
    return _read_hdf5_file(download_glove_200, GLOVE_200_ANGULAR_FILE, 200, MetricType.COSINE)


def download_mnist() -> None:
    _download_hdf5_file(MNIST_784_EUCLIDEAN_FILE, MNIST_784_EUCLIDEAN_DOWNLOAD_URL)


def read_mnist() -> Dataset:
    return _read_hdf5_file(download_mnist, MNIST_784_EUCLIDEAN_FILE, 784, MetricType.L2)


def download_fashion_mnist() -> None:
    _download_hdf5_file(FASHION_MNIST_784_EUCLIDEAN_FILE, FASHION_MNIST_784_EUCLIDEAN_DOWNLOAD_URL)


def read_fashion_mnist() -> Dataset:
    return _read_hdf5_file(download_fashion_mnist, FASHION_MNIST_784_EUCLIDEAN_FILE, 784, MetricType.L2)


def download_deep_image():
    _download_hdf5_file(DEEP_IMAGE_96_ANGULAR_FILE, DEEP_IMAGE_96_ANGULAR_DOWNLOAD_URL)


def read_deep_image() -> Dataset:
    return _read_hdf5_file(download_deep_image, DEEP_IMAGE_96_ANGULAR_FILE, 96, MetricType.COSINE)


def _download_tar_file(file_name: str, url: str):
    if not os.path.exists(os.path.join(DATA_BASE_PATH, file_name)):
        download(url, os.path.join(DATA_BASE_PATH, file_name))
        os.system(f"tar -xvf {os.path.join(DATA_BASE_PATH, file_name)} -C {DATA_BASE_PATH}")
        os.remove(os.path.join(DATA_BASE_PATH, file_name))


def _read_tar_file(download_func: Callable[[], None], name: str, dimension: int, metric_type: MetricType) -> Dataset:
    download_func()
    return Dataset(dimension, metric_type,
                   fvecs_read(os.path.join(DATA_BASE_PATH, f"{name}/{name}_base.fvecs")),
                   fvecs_read(os.path.join(DATA_BASE_PATH, f"{name}/{name}_query.fvecs")),
                   ivecs_read(os.path.join(DATA_BASE_PATH, f"{name}/{name}_groundtruth.ivecs")))


def _download_hdf5_file(file_name: str, url: str):
    if not os.path.exists(os.path.join(DATA_BASE_PATH, file_name)):
        download(url, os.path.join(DATA_BASE_PATH, file_name))


def _read_hdf5_file(download_func: Callable[[], None], file_name: str, dimension: int,
                    metric_type: MetricType) -> Dataset:
    download_func()
    with h5py.File(os.path.join(DATA_BASE_PATH, file_name), 'r') as f:
        return Dataset(dimension, metric_type,
                       f["train"][()].tolist(),
                       f["test"][()].tolist(),
                       f["neighbors"][()].tolist())
