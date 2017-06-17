from tensorflow.python.ops.rnn_cell_impl import RNNCell
from EUNN import *



def modReLU(z, b, comp):
	if comp:
		return z
		z_norm = math_ops.sqrt(math_ops.square(math_ops.real(z)) + math_ops.square(math_ops.imag(z))) + 0.00001
		step1 = nn_ops.bias_add(z_norm, b)
		step2 = math_ops.complex(nn_ops.relu(step1), array_ops.zeros_like(z_norm))
		step3 = z/math_ops.complex(z_norm, array_ops.zeros_like(z_norm))
	else:
		z_norm = math_ops.abs(z) + 0.00001
		step1 = nn_ops.bias_add(z_norm, b)
		step2 = nn_ops.relu(step1)
		step3 = math_ops.sign(z)
		
	return math_ops.multiply(step3, step2)




class EURNNCell(RNNCell):
	"""Efficient Unitary Recurrent Network Cell
	The implementation is based on: http://arxiv.org/abs/1612.05231.

	"""

	def __init__(self, hidden_size, capacity=2, FFT=False, comp=False, activation=modReLU):
		
		self._hidden_size = hidden_size
		self._activation = activation
		self._capacity = capacity
		self._FFT = FFT
		self._comp = comp


		self.v1, self.v2, self.ind, self.diag, self._capacity = EUNN_param(hidden_size, capacity, FFT, comp)



	@property
	def state_size(self):
		return self._hidden_size

	@property
	def output_size(self):
		return self._hidden_size

	@property
	def capacity(self):
		return self._capacity

	def __call__(self, inputs, state, scope=None):
		with vs.variable_scope(scope or "eurnn_cell"):

			Wh = EUNN_loop(state, self._capacity, self.v1, self.v2, self.ind, self.diag)

			U_init = init_ops.random_uniform_initializer(-0.01, 0.01)
			if self._comp:
				U_re = vs.get_variable("U_re", [inputs.get_shape()[-1], self._hidden_size], initializer = U_init)
				U_im = vs.get_variable("U_im", [inputs.get_shape()[-1], self._hidden_size], initializer = U_init)
				Ux_re = math_ops.matmul(inputs, U_re)
				Ux_im = math_ops.matmul(inputs, U_im)
				Ux = math_ops.complex(Ux_re, Ux_im)
			else:
				U = vs.get_variable("U", [inputs.get_shape()[-1], self._hidden_size], initializer = U_init)
				Ux = math_ops.matmul(inputs, U) 

			bias = vs.get_variable("modReLUBias", [self._hidden_size], initializer= init_ops.constant_initializer())
			output = self._activation((Ux + Wh), bias, self._comp)  

		return output, output

