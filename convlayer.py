import layerclass
import scipy.ndimage
import numpy as np
import op_funcs as of
class conv_layer(layerclass.Layer):
	def __init__(self, dim, kernels):
		self.dim = dim
		self.kernels = kernels
		self.wm = .005 *np.random.randn(self.dim, self.dim, kernels)
		self.b = np.random.randn(1, kernels)
	def forward_prop(self, image):
		ndim = image.shape[0] - self.dim + 1
		convolved_matrix = np.zeros((ndim, ndim, self.kernels))
		convolved_matrix = of.convolve(image, self.wm, self.kernels, convolved_matrix) 
		convolved_matrix = of.sigmoid(convolved_matrix + self.b)
		return convolved_matrix
	def back_prop(self):
		return super(conv_layer, self).back_prop()