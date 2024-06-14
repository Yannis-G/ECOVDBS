import os.path
import sys
from urllib.request import urlretrieve

import numpy as np

from client.base_config import MetricType
from client.chroma_client import ChromaClient
from client.milvus_client import MilvusClient
from client.milvus_config import MilvusAutoIndexConfig, MilvusHNSWConfig
from client.redis_client import RedisClient
from client.redis_config import RedisHNSWConfig
from client.pgvector_client import PgvectorClient
from client.pgvector_config import PgvectorHNSWConfig


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


# Datasource http://corpus-texmex.irisa.fr/
if __name__ == '__main__':
    # Example usage:
    filename_gt = os.path.join(os.path.dirname(__file__), "../data/siftsmall/siftsmall_groundtruth.ivecs")
    filename_b = os.path.join(os.path.dirname(__file__), "../data/siftsmall/siftsmall_base.fvecs")
    filename_q = os.path.join(os.path.dirname(__file__), "../data/siftsmall/siftsmall_query.fvecs")

    # Read files into arrays
    v_gt = ivecs_read(filename_gt)
    v_b = fvecs_read(filename_b)
    v_q = fvecs_read(filename_q)
    v_gt = v_gt.tolist()
    v_b = v_b.tolist()
    v_q = v_q.tolist()

    # Initialize ChromaClient and insert data
    client = ChromaClient(128)
    client.create_index()
    client.insert(v_b)

    # Query and evaluate results
    res = client.query(v_q[0], 100)
    c = 0
    for id in res:
        if v_gt[0].__contains__(int(id)):
            c += 1
    print(c)
    v_gt[0].sort()
    res.sort()
    print(v_gt[0])
    print(res)
    print(client.disk_storage())
    print(client.index_storage())

# Test for reading hdf5 file
#    with h5py.File(os.path.join(os.path.dirname(__file__), "../data/glove-25-angular.hdf5", 'r') as f:
#
#        # Access the dataset directly (assuming no subgroups)
#        dset: h5py.Dataset = f['test']
#        print(list(dset))
#        print(dset[2].nbytes)
#        sum = 0
#        for key in f.keys():
#            dset: h5py.Dataset = f[key]
#            sum += dset.nbytes
#            print(f"Key: {key} Size: {dset.nbytes}")
#        print(sum)
