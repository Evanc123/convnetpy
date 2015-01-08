import maxpoollayer as mp
import fullyconnectedlayer as fcl
import convlayer as cl
from numpy import arange
import op_funcs as of
import numpy as np
from numpy import *
from numbapro import cuda, float32
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy import signal, misc

X = np.load("mnistdata.npy")
y = np.load("mnisttarget.npy")
#image = X[19999, :].reshape(28, 28)

X_train, y_train = X[:70000] / 255., y[:70000]

X_train, y_train = shuffle(X_train, y_train)

X_test, y_test = X_train[60000:70000], y_train[60000:70000]

X_train, y_train = X_train[:60000], y_train[:60000]

epochs = 3
mini_batch_size = 1
input_channels = 1

conv1 = cl.conv_layer(6, 5, 24, mini_batch_size, 1)# Kernels, size, output_size, mini_batch
maxp1 = mp.max_pool_layer(2, 12, mini_batch_size)
conv2 = cl.conv_layer(50, 5, 8, mini_batch_size, 6)
maxp2 = mp.max_pool_layer(2, 4, mini_batch_size)
fcl1 = fcl.fully_connected_layer(800, 500 )
fcl2 = fcl.fully_connected_layer(500, 10)
from numpybpnn import *
NN = NN(800, 500, 10)
error = 0
for i in range(epochs):
	len_div_batch_size = X_train.shape[0] / mini_batch_size
	for j in range(len_div_batch_size):
		images = np.zeros((mini_batch_size, input_channels, 28, 28))
		answers = np.zeros((mini_batch_size, 10))
		for i in range(mini_batch_size):
			for c in range(input_channels):
				images[i][c] = X_train[i + (j * mini_batch_size), :].reshape(28, 28)
				answers[i][y_train[i + j * mini_batch_size]] = 1

		o_conv1 = conv1.forward_prop(images)
		o_maxp1 = maxp1.forward_prop(o_conv1)


		o_conv2 = conv2.forward_prop(o_maxp1)

		o_maxp2 = maxp2.forward_prop(o_conv2)

		#o_fcl1 = fcl1.forward_prop(o_maxp2, softmax = False)
		pat = (o_maxp2.T.ravel().T.ravel(), answers[0])
		#o_fcl2 = fcl2.forward_prop(o_fcl1, softmax=True)
		#print o_fcl2

		for p in range(1):
			input = pat[0]
			target = pat[1]
			NN.update(input)
			error = error + NN.backPropagate(target, 0.5, .1)
		print error

		#batch_cost = np.zeros((mini_batch_size, 1))
		#for i in range(mini_batch_size):
		#	batch_cost[i] = (o_fcl2[i] - answers[i] ).sum()
		#cost_of_batch = (1/mini_batch_size) * batch_cost.sum()
		#print cost_of_batch
		#d_ofcl2 = fcl2.back_prop(o_fcl2)
		#print #d_ofcl2.shape

	print len_div_batch_size, 'iteration number'




# #print y.shape
# images = np.zeros((mini_batch, og_channels, 28, 28))
# answers = np.zeros((mini_batch, 10))
# for i in range(mini_batch):
#
# 	for c in range(og_channels):
# 		images[i][c] = X_train[i, :].reshape(28, 28)
# 		answers[i][y_train[i]] = 1
#
#
#
# conv1 = cl.conv_layer(6, 5, 24, mini_batch, 1)# Kernels, size, output_size, mini_batch
#
# maxp1 = mp.max_pool_layer(2, 12, mini_batch)
# conv2 = cl.conv_layer(50, 5, 8, mini_batch, 6)
# maxp2 = mp.max_pool_layer(2, 4, mini_batch)
# n = NN(800, 500, 10)
# n.train(pat)
# #fcl1 = fcl.fully_connected_layer(800, 500)
# #fcl2 = fcl.fully_connected_layer(500, 10)
#
#
#
# o_conv1 = conv1.forward_prop(images)
# o_maxp1 = maxp1.forward_prop(o_conv1)
#
# o_conv2 = conv2.forward_prop(o_maxp1)
# o_maxp2 = maxp2.forward_prop(o_conv2)
#
# o_fcl1 = fcl1.forward_prop(o_maxp2)
# o_fcl2 = fcl2.forward_prop(o_fcl1)
# softmax_out = of.softmax(o_fcl2, 1.0)
#
#
# batch_cost = np.zeros((mini_batch, 1))
# for i in range(mini_batch):
# 	batch_cost[i] = of.cost(answers[i], softmax_out[i])
# print batch_cost




#plt.figure()
#plt.imshow(image)
#plt.gray()
#plt.show()












