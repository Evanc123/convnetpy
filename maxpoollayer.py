import layerclass
import numpy as np
import op_funcs as of
class max_pool_layer(layerclass.Layer):
	def __init__(self, dim, output_dim, mini_batch):
		self.dim = dim
		self.output_dim = output_dim
		self.mini_batch = mini_batch
	def forward_prop(self, input_maps):
		# batch, channels, #kernels, outputsize, outputsize
		#print input_maps[0][0],'tesafdsa', input_maps[1][0]
		input_dim = input_maps.shape
		self.input_maps = input_maps
		#TODO vectorize maxpooling
		output_map = np.zeros((self.mini_batch, input_dim[1], input_dim[2] / self.dim, input_dim[3] /self.dim))
		
		#print output_map.shape, input_maps.shape
		for i in range(self.mini_batch):
			
			for k in range(input_dim[1]): #kernels
				rowcount = 0
				for row in xrange(0, input_dim[2], self.dim):
					colcount = 0
					for col in xrange(0, input_dim[2], self.dim):
						output_map[i,  k,rowcount ,colcount] = input_maps[i,  k, row:(row + self.dim),col:(col+self.dim)].max()
						colcount +=1
					rowcount +=1
		return output_map
	def back_prop(self, deltasl_1):
		return 
		pass