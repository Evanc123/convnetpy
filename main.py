import maxpoollayer as mp
import fullyconnectedlayer as fcl
import convlayer as cl
from numpy import arange

import numpy as np




image = np.random.randn(32, 32)

cl1 = cl.conv_layer(5, 6)
mp1 = mp.max_pool_layer(3)


icv1 = cl1.forward_prop(image)
imp1 = mp1.forward_prop(icv1)
print imp1


 


