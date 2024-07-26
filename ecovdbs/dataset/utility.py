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
# Modifications: Moved ivecs_read in a separate method and added a tolist() call to the return value.

def _ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def ivecs_read(fname):
    return _ivecs_read(fname).tolist()


def fvecs_read(fname):
    return _ivecs_read(fname).view('float32').tolist()
