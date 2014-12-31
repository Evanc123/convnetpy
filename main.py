import maxpoollayer as mp
import fullyconnectedlayer as fcl
import convlayer as cl
from numpy import arange
import op_funcs as of
import numpy as np
from numpy import *
from numbapro import cuda, float32
import matplotlib.pyplot as plt
from scipy import signal, misc

X = np.load("mnistdata.npy")
y = np.load("mnisttarget.npy")
image = X[0, :].reshape(28, 28)

of.elementmultiply.max_blocksize = 32


cl1 = cl.conv_layer(7, kernels = 5)
mp1 = mp.max_pool_layer(3)
cl2 = cl.conv_layer(5, kernels = 25)
mp2 = mp.max_pool_layer(3)
fc1 = fcl.fully_connected_layer(784) # 4 features
fc2 = fcl.fully_connected_layer(10)


out_cl1 = cl1.forward_prop(image)
out_mp1 = mp1.forward_prop(out_cl1)
out_cl2 = cl2.forward_prop(out_mp1)
out_mp2 = mp2.forward_prop(out_cl2)




#ifc1 = fc1.forward_prop(imp1)


plt.figure()
plt.imshow(out_mp2[:, :, 0])
plt.gray()
plt.show()




"""
a = np.random.randn(300, 300)
b = np.random.randn(300, 300)
c = np.random.randn(300, 300)
print of.elementmultiply(a, b)
"""






