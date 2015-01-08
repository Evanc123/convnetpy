import layerclass
import scipy.ndimage
import numpy as np
import op_funcs as of 
class conv_layer(layerclass.Layer):
	def __init__(self, kernels, dim, output_size, mini_batch_size, channels):
		self.dim = dim
		self.num_kernels = kernels
		self.channels = channels
		self.epsilon = .5
		self.kernels = np.random.randn(self.channels, kernels, dim, dim) #* 2 * self.epsilon - self.epsilon# channel x 6 x 5 x 5 weights
		self.output_size = output_size

		self.biases = np.random.randn( kernels, output_size, output_size) #* 2 * self.epsilon - self.epsilon
		self.mini_batch = mini_batch_size
		self.deltas = np.zeros((self.mini_batch, self.num_kernels, dim, dim))
	def forward_prop(self, inp):
		
		#print np.array_equal(inp[0][0], inp[1][0])
		self.inp = inp
		self.output_maps = np.zeros((self.mini_batch, self.num_kernels, self.output_size, self.output_size))
		
		
		self.pre_activated_outputs = of.conv_forward_prop(inp, self.kernels)

		#np.array_equal(self.pre_activated_outputs[0][0], self.pre_activated_outputs[1][0])
		#print np.array_equal(self.pre_activated_outputs[0][0], self.pre_activated_outputs[1][0])
		for i in range(self.mini_batch):
			for k in range(self.num_kernels):
				for c in range(self.channels):
					self.output_maps[i][k] = (of.tanh(self.pre_activated_outputs[i][k] + self.biases[k]))
		

		#print np.array_equal(self.output_maps[0][0], self.output_maps[1][0])
		return self.output_maps
		
	def back_prop(self, d_lplus1): #assuming d_lplus1 is mini_batch x 6 x 5 x 5
		for i in range(self.mini_batch):
			for j in range(self.num_kernels):
				deltas[i][j] = of.tanh_prime(kernels[j]) * d_lplus1[i][j]
		