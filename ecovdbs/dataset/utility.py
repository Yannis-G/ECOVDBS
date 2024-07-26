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


# The following IO/eval functions are from faiss
# https://github.com/facebookresearch/faiss/blob/master/benchs/datasets.py

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')
