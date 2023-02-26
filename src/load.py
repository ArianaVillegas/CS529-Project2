import csv
import array
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz, load_npz


def read_large_df(filename, chunksize=1000):
    """
    Utility that read large csv in chunks and return a dataframe, 
    if a npz file is found, the npz file is loaded instead.
    
    Parameters
    ----------
    filename : path to the csv file.
    
    Returns
    -------
    df: sparse array read from npz file.
    """
    
    npz_file = filename[:-3] + 'npz'
    if not Path(npz_file).is_file():
        data = array.array("f")
        indices = array.array("i")
        indptr = array.array("i", [0])
        with open(filename, 'r') as file:
            for i, row in enumerate(csv.reader(file), 1):
                row = np.array(list(map(float, row)))
                shape1 = len(row)
                nonzero = np.where(row)[0]
                data.extend(row[nonzero])
                indices.extend(nonzero)
                indptr.append(indptr[-1]+len(nonzero))
        sparse_matrix = csr_matrix((data, indices, indptr),
                            dtype=float, shape=(i, shape1))
        save_npz(npz_file, sparse_matrix)
    sparse_matrix = load_npz(npz_file)
    return sparse_matrix