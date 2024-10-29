import numpy as np

cimport cython
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def ccm(np.ndarray[DTYPE_t, ndim=2] test_signal, np.ndarray[DTYPE_t, ndim=2] ref_signal, distance_metric):
    num_test_vectors = test_signal.shape[0]
    num_test_features = test_signal.shape[1]

    xy = np.dot(test_signal, ref_signal.T)

    x_squared = np.square(test_signal).sum(axis=1)
    y_squared = np.square(ref_signal).sum(axis=1)
    cost_matrix = (x_squared + (-2 * xy + y_squared).T).T

    if distance_metric != 'euclidean_squared':
        # don't sqrt if using square of euclidean distance
        cost_matrix = np.sqrt(cost_matrix)

    # normalise with the number of vector features
    cost_matrix /= num_test_features

    cost_matrix = cost_matrix.astype(np.float64)

    return cost_matrix
