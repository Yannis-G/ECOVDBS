import os

from .utility import download

from ..config import DATA_BASE_PATH


def download_sift_small():
    if not os.path.exists(os.path.join(DATA_BASE_PATH, "siftsmall")):
        download("ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
                 os.path.join(DATA_BASE_PATH, "siftsmall.tar.gz"))
        os.system(f"tar -xvf {os.path.join(DATA_BASE_PATH, 'siftsmall.tar.gz')} -C {DATA_BASE_PATH}")
        os.remove(os.path.join(DATA_BASE_PATH, "siftsmall.tar.gz"))


def download_sift():
    if not os.path.exists(os.path.join(DATA_BASE_PATH, "sift")):
        download("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
                 os.path.join(DATA_BASE_PATH, "sift.tar.gz"))
        os.system(f"tar -xvf {os.path.join(DATA_BASE_PATH, 'sift.tar.gz')} -C {DATA_BASE_PATH}")
        os.remove(os.path.join(DATA_BASE_PATH, "sift.tar.gz"))


def download_glove():
    if not os.path.exists(os.path.join(DATA_BASE_PATH, "glove-25-angular.hdf5")):
        download("https://ann-benchmarks.com/glove-25-angular.hdf5",
                 os.path.join(DATA_BASE_PATH, "glove-25-angular.hdf5"))


def download_fashion_mnist():
    if not os.path.exists(os.path.join(DATA_BASE_PATH, "fashion-mnist-784-euclidean.hdf5")):
        download("https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
                 os.path.join(DATA_BASE_PATH, "fashion-mnist-784-euclidean.hdf5"))
