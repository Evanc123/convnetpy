import layerclass
import scipy.ndimage
import numpy as np
import op_funcs as of
class conv_layer(layerclass.Layer):
	def __init__(self, dim, kernels):
		self.dim = dim
		self.kernels = kernels
		self.wm = np.random.randn(self.dim, self.dim, kernels)
		self.b = np.random.randn(1, kernels)
	def forward_prop(self, image):
		newdim = image.shape[0] - self.dim + 1
		new_matrix = np.zeros((newdim, newdim, self.kernels))
		new_matrix = of.convolve(image, self.wm, self.kernels, new_matrix) 
		new_matrix = of.sigmoid(new_matrix + self.b)

		return new_matrix
	def back_prop(self):
		return super(conv_layer, self).back_prop()