

from numbapro import vectorize, jit
from numpy import arange
from numbapro import guvectorize
from numpy import arange
from numbapro import autojit, jit
from numbapro import cuda, float32, void, f8, i2
import numpy as np

@vectorize([float32(float32, float32, float32)], target = 'gpu')
def vec_saxpy(a, x, y):
	return a * x + y




@guvectorize(['void(float64[:,:], float64[:,:], float64[:,:])'],
             '(m,n),(n,p)->(m,p)')
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
				new_matrix[i][j][k] = (image[i:(len_weights + i), j:(len_weights + j)] * weights[:, :, k]).sum()
	return new_matrix

@autojit
def sigmoid(A):
	return 1.0 / (1.0 + np.exp(-A))
@autojit
def maxnum(A):
	return np.amax(A)