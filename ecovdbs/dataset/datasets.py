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


def fvecs_read(filename, bounds=None):
    """
    Read a file in fvecs format. This code is a python translation of http://corpus-texmex.irisa.fr/fvecs_read.m

    :param filename: Path to the fvecs file.
    :param bounds: Bounds to read a subset of vectors.
    :return: Array of vectors.
    """
    return __xvecs_read(filename, np.float32, bounds)


def ivecs_read(filename, bounds=None):
    """
    Read a file in ivecs format. This code is a python translation of http://corpus-texmex.irisa.fr/ivecs_read.m

    :param filename: Path to the ivecs file.
    :param bounds: Bounds to read a subset of vectors.
    :return: Array of vectors.
    """
    return __xvecs_read(filename, np.int32, bounds)


def __xvecs_read(filename, dtype: np.int32 | np.float32, bounds=None):
    """
    Read a file in ivecs(dtype=np.int32) or fvecs(dtype=np.float32) format. This code is a python translation of
    http://corpus-texmex.irisa.fr/ivecs_read.m and http://corpus-texmex.irisa.fr/fvecs_read.m

    :param filename: Path to the ivecs file.
    :param dtype: Data type of the vectors.
    :param bounds: Bounds to read a subset of vectors.
    :return: Array of vectors.
    """
    # Open the file
    with open(filename, 'rb') as fid:
        # Read the vector size
        d = np.fromfile(fid, dtype=np.int32, count=1)[0]
        vecsizeof = 1 * 4 + d * 4

        # Get the number of vectors
        fid.seek(0, 2)
        bmax = fid.tell() // vecsizeof
        a, b = 1, bmax

        if bounds is not None:
            if len(bounds) == 1:
                b = bounds[0]
            elif len(bounds) == 2:
                a, b = bounds

        assert a >= 1
        if b > bmax:
            b = bmax

        if b == 0 or b < a:
            return np.array([])

        # Compute the number of vectors that are really read and go in starting positions
        n = b - a + 1
        fid.seek((a - 1) * vecsizeof, 0)

        # Read n vectors
        v = np.fromfile(fid, dtype=dtype, count=(d + 1) * n)
        v = np.reshape(v, (d + 1, n), order='F')

        # Check if the first column (dimension of the vectors) is correct
        assert np.sum(v[0, 1:] == v[0, 0]) == n - 1
        v = v[1:, :]

    return v.T  # Transpose to have each vector stored in a row
