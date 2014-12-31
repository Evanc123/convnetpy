import layerclass
import numpy as np
import op_funcs as of
class max_pool_layer(layerclass.Layer):
	def __init__(self, dim):
		self.dim = dim
		
	def forward_prop(self, image):
		ndim = image.shape[0] - self.dim + 1
		nmatrix = np.zeros((ndim, ndim, image.shape[2])) # assumes [2] is kernels

		return nmatrix.shape
	def back_prop(self):
		pass