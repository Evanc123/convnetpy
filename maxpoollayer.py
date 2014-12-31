import layerclass
import numpy as np
import op_funcs as of
class max_pool_layer(layerclass.Layer):
	def __init__(self, dim):
		self.dim = dim
		
	def forward_prop(self, feature_maps):
		ndim = feature_maps.shape[0] - self.dim + 1
		pooled_feature_maps = np.zeros((ndim, ndim, feature_maps.shape[2])) # assumes [2] is kernels
		kernels = feature_maps.shape[2]
		pooled_feature_maps = of.maxpool(feature_maps, kernels, pooled_feature_maps, self.dim)
		
		return pooled_feature_maps
	def back_prop(self):
		pass