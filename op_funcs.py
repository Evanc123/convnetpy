

import matplotlib.pyplot as plt
from numbapro import vectorize, jit, guvectorize, autojit, cuda, float32, void, f8, i2
import numpy as np
from numpy import arange
import math
import numpy as np
import scipy.signal as sc
from numba import autojit
import scipy
from sklearn import metrics

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

@jit(f8[:, :, :](f8[:, :, :], i2, f8[:, :, :], i2)) #rewrite so it is not "valid"
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
def conv2(A, B, m ):
    # 'valid', 'same', 'full'
    return sc.fftconvolve(A, B, mode = m)

@autojit
def kron(A, B):
    #B is size of max pool
    return np.kron(A, B)
@autojit
def rot180(A):
    return scipy.ndimage.interpolation.rotate(A, 180)
@autojit
def rot90(A):
    return np.rot90(A)
@autojit
def softplus(A):
    return np.log(1 + np.exp(A))
@autojit
def sofltplusprime(A):
    return 1 / (1 + np.exp(-A))
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

@autojit
def tanh(A):
    return np.tanh(A)

@autojit
def tanh_prime(A):
    return 1.0 - np.exp2(A)

#not returning distinct arrays
@autojit
def conv_forward_prop(inp, kernels):
    
    new_dim = inp.shape[2] - kernels.shape[3] + 1
    #print inp.shape
    #print inp.shape[0] is batch size, 1 is # of channels, 2 and 3 are dim
    #print kernels.shape[0] is number of channels, 1 is number of kernels, 2 and 3 are dim
    batch_size, num_channels, num_kernels = inp.shape[0], kernels.shape[0], kernels.shape[1]
    #plt.figure()
    
    pre_activated = np.zeros((batch_size, num_kernels, new_dim, new_dim)) #looks right
    for i in range(batch_size): # foreach image 
        for k in range(num_kernels):
            for c in range(num_channels):
                # plt.imshow(kernels[c][k])
                # plt.gray()
                # plt.show()
                pre_activated[i][k] += conv2(rot90(inp[i][c]), kernels[c][k], 'valid')
    return pre_activated
    
@autojit
def cost(y_true, y_pred):
    return metrics.log_loss(y_true, y_pred)
@autojit
def CrossEntropyCost(a, y):
	return np.nan_to_num(np.sum(-y * np.log(a)-(1-y)*np.log(1-a)))