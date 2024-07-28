import os.path
from urllib.request import urlretrieve

import numpy as np


def download(src_url: str, dest_path: str) -> None:
    """
    Download a file from a source URL to a destination path if it doesn't exist.

    :param src_url: Source URL of the file to download.
    :param dest_path: Destination path to save the downloaded file.
    """
    if not os.path.exists(dest_path):
        urlretrieve(src_url, dest_path)


# The following method is copied from Faiss.
# Original Author: Lucas Hosseini
# Source: https://github.com/facebookresearch/faiss/blob/master/benchs/datasets.py
# Modifications: Moved ivecs_read in a separate method, added a tolist() call to the return value and add docstring.

def _ivecs_read(fname):
    """
    Read an .ivecs file and return its contents as a numpy array.

    :param fname: Path to the .ivecs file.
    :return: Numpy array of the file contents.
    """
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def ivecs_read(fname):
    """
    Read an .ivecs file and return its contents as a list.

    :param fname: Path to the .ivecs file.
    :return: List of vectors from the .ivecs file.
    """
    return _ivecs_read(fname).tolist()


def fvecs_read(fname):
    """
    Read an .fvecs file and return its contents as a list.

    :param fname: Path to the .fvecs file.
    :return: List of vectors from the .fvecs file.
    """
    return _ivecs_read(fname).view('float32').tolist()
