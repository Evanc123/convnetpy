import layerclass
import numpy as np
class fully_connected_layer(layerclass.Layer):
	def __init__(self, num_neurons):
		self.num_neurons = num_neurons
		self.ni = num_neurons + 1
		self.nh = num_neurons
		self.ai = np.ones(self.ni)
		self.ah = np.ones(self.nh)
		self.wi = np.random.rand(self.ni, self.nh)
	def forward_prop(self, feature_maps):
		#choose random 4 feature maps
		input = feature_maps[:, :,  
		self.ai = np.append(feature_maps, 1)
		self.ah = of.sigmoid((self.ai * self.wi.transpose()).sum(axis = 1))
		
		 
	def back_prop(self, input):
		pass