


from numbapro import vectorize, jit, guvectorize, autojit, cuda, float32, void, f8, i2
import numpy as np
from numpy import arange
import math


@vectorize([float32(float32, float32, float32)], target = 'gpu')
def vec_saxpy(a, x, y):
	return a * x + y

@guvectorize(['void(float64[:,:], float64[:,:], float64[:,:])'],
             '(m,n),(n,p)->(m,p)', target = 'gpu')
def matmul(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

@jit(f8[:, :, :](f8[:, :], f8[:, :, :], i2, f8[:,:,:]))
def convolve(image, weights, kernels, new_matrix):
	ndim = image.shape[0] - weights.shape[0] + 1
	len_weights = weights.shape[0]
	for k in range(0, kernels):
		for i in range(0, ndim):
			for j in range(0 , ndim):
				new_matrix[i][j][k] =  (image[i:(len_weights + i), j:(len_weights + j)] * weights[:, :, k]).sum() # elementmultiply(image[i:(len_weights + i), j:(len_weights + j)], weights[:, :, k]).sum()
	return new_matrix

@jit(f8[:, :, :](f8[:, :, :], i2, f8[:, :, :], i2))
def maxpool(feature_maps, kernels, pooled_feature_maps, poolsize):
	ndim = feature_maps.shape[0] - poolsize + 1
	for k in range(kernels):
		for i in range(ndim):
			for j in range(ndim):
				pooled_feature_maps[i][j][k] = feature_maps[i:(ndim + i), j:(ndim + j), k].max()
	return pooled_feature_maps

@autojit
def sigmoid(A):
	return 1.0 / (1.0 + np.exp(-A))


@guvectorize(['void(int32[:,:], int32[:,:], int32[:,:])',
              'void(float64[:,:], float64[:,:], float64[:,:])'],
              '(x, y),(x, y)->(x, y)', target = 'gpu')
def elementmultiply(a, b, c):
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            c[i, j] = a[i, j] * b[i, j]




@autojit
def softmax(w, t=1.0):
    """Calculate the softmax of a list of numbers w.
    @param w: list of numbers
    @return a list of the same length as w of non-negative numbers
    >>> softmax([0.1, 0.2])
    array([ 0.47502081,  0.52497919])
    >>> softmax([-0.1, 0.2])
    array([ 0.42555748,  0.57444252])
    >>> softmax([0.9, -10])
    array([  9.99981542e-01,   1.84578933e-05])
    >>> softmax([0, 10])
    array([  4.53978687e-05,   9.99954602e-01])
    """
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist
