import numpy as np

def np_one_hot( v, n ):

    """
        Encode v into one hot vectors of dimension n

        param v: int 1d array
        param n: int, dimension of resulting vectors
    """

    v = v.astype(int)
    arr = np.zeros(shape=(len(v),n))
    arr[np.arange(len(v)), v] = 1.
    return arr
