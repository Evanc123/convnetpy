import layerclass
import numpy as np
import op_funcs as of
class fully_connected_layer(layerclass.Layer):
	def __init__(self, l_in, l_out):
		self.l_in = l_in
		self.l_out = l_out
		self.epsilon = .5
		self.weights = np.random.randn(l_in, l_out) #* 2 - self.epsilon + self.epsilon
		self.biases = np.random.randn(l_out) #* 2 - self.epsilon + self.epsilon
		
	def forward_prop(self, feature_maps, softmax):
		#FCL's not getting distinct values
		#print feature_maps[0][0], 'testasetsdfasd', feature_maps[1][0]
		self.in_maps = feature_maps
		self.pre_a_output = np.zeros((feature_maps.shape[0], self.l_out))

		for i in range(feature_maps.shape[0]):

			inp = feature_maps[i].T.ravel().T.ravel() # batch image
			self.pre_a_output[i] = np.dot(self.weights.T, inp) + self.biases
		#print self.pre_a_output[0].shape, self.pre_a_output.shape
		self.activated_output = np.zeros((feature_maps.shape[0], self.l_out))
		if (softmax == True):
			self.activated_output = of.softmax(self.pre_a_output, 1.0)
		else:
			for i in range(feature_maps.shape[0]):
				self.activated_output[i] = np.tanh(self.pre_a_output[i])
		return self.activated_output

		
		 
	def back_prop(self, softmax_out):
		output_deltas = of.tanh_prime(self.activated_output) * softmax_out
		N = .01
		for j in range(self.l_in):
			for k in range(self.l_out):
				change = output_deltas[k] * self.in_maps[j]
				self.weights[j][k] = self.weights[j][k] * N * change
				self.biases[j][k] = self.biases[j][k] * N * output_deltas[k]



		return output_deltas