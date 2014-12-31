from abc import ABCMeta, abstractmethod
import numpy as np
import numba_funcs
class Layer(object): 
	__metaclass__ = ABCMeta
	@abstractmethod
	def forward_prop(self):
		raise NotImplementedError( "Should have implemented this" )
	@abstractmethod
	def back_prop(self):
		raise NotImplementedError( "Should have implemented this" )