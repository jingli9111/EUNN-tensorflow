from tensorflow.contrib.rnn.python.ops import *

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

def toTensorArray(elems):
	elems = ops.convert_to_tensor(elems)
	n = array_ops.shape(elems)[0]
	elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, 
											size=n,
											dynamic_size=False,
											infer_shape=True, 
											clear_after_read = False)
	elems_ta = elems_ta.unstack(elems)
	return elems_ta


class EURNNCell(core_rnn_cell.RNNCell):


	def __init__(self, hidden_size, capacity, FFT, comp, activation=modReLU):
		
		self._hidden_size = hidden_size
		self._activation = activation
		self._capacity = capacity
		self._FFT = FFT
		self._comp = comp


		theta_phi_initializer = init_ops.random_uniform_initializer(-np.pi, np.pi)
		if self._FFT:
			self._capacity = int(np.log2(hidden_size))

			params_theta_0 = vs.get_variable("theta_0", [self._capacity, hidden_size/2], initializer=theta_phi_initializer)
			cos_theta_0 = math_ops.cos(params_theta_0)
			sin_theta_0 = math_ops.sin(params_theta_0)
			
			if self._comp:

				params_phi_0 = vs.get_variable("phi_0", [self._capacity, hidden_size/2], initializer=theta_phi_initializer)
				cos_phi_0 = math_ops.cos(params_phi_0)
				sin_phi_0 = math_ops.sin(params_phi_0)

				cos_list_0_re = array_ops.concat([cos_theta_0, math_ops.multiply(cos_theta_0, cos_phi_0)], 1)
				cos_list_0_im = array_ops.concat([array_ops.zeros_like(cos_theta_0), math_ops.multiply(cos_theta_0, sin_phi_0)], 1)
				sin_list_0_re = array_ops.concat([sin_theta_0, -math_ops.multiply(sin_theta_0, cos_phi_0)], 1)
				sin_list_0_im = array_ops.concat([array_ops.zeros_like(sin_theta_0), -math_ops.multiply(sin_theta_0, sin_phi_0)], 1)
				cos_list_0 = array_ops.unstack(math_ops.complex(cos_list_0_re, cos_list_0_im))
				sin_list_0 = array_ops.unstack(math_ops.complex(sin_list_0_re, sin_list_0_im))

			else:
				cos_list_0 = array_ops.unstack(array_ops.concat([cos_theta_0, cos_theta_0], 1))
				sin_list_0 = array_ops.unstack(array_ops.concat([sin_theta_0, -sin_theta_0], 1))

			

			ind, ind1 = permute_FFT(hidden_size)
			ind1_list = array_ops.unstack(ind1)


			diag_list_0 = []
			off_list_0 = []
			for i in range(self._capacity):
				diag_list_0.append(permute(cos_list_0[i], ind1_list[i]))
				off_list_0.append(permute(sin_list_0[i], ind1_list[i]))
			v1 = array_ops.stack(diag_list_0, 0)
			v2 = array_ops.stack(off_list_0, 0)

		else:

			params_theta_0 = vs.get_variable("theta_0", [self._capacity/2, hidden_size/2], initializer=theta_phi_initializer)
			cos_theta_0 = math_ops.cos(params_theta_0)
			sin_theta_0 = math_ops.sin(params_theta_0)
			if self._comp:

				params_phi_0 = vs.get_variable("phi_0", [self._capacity/2, hidden_size/2], initializer=theta_phi_initializer)
				cos_phi_0 = math_ops.cos(params_phi_0)
				sin_phi_0 = math_ops.sin(params_phi_0)

				cos_list_0_re = array_ops.concat([cos_theta_0, math_ops.multiply(cos_theta_0, cos_phi_0)], 1)
				cos_list_0_im = array_ops.concat([array_ops.zeros_like(cos_theta_0), math_ops.multiply(cos_theta_0, sin_phi_0)], 1)
				sin_list_0_re = array_ops.concat([sin_theta_0, -math_ops.multiply(sin_theta_0, cos_phi_0)], 1)
				sin_list_0_im = array_ops.concat([array_ops.zeros_like(sin_theta_0), -math_ops.multiply(sin_theta_0, sin_phi_0)], 1)
				cos_list_0 = array_ops.unstack(math_ops.complex(cos_list_0_re, cos_list_0_im))
				sin_list_0 = array_ops.unstack(math_ops.complex(sin_list_0_re, sin_list_0_im))


			else:
				cos_list_0 = array_ops.concat([cos_theta_0, cos_theta_0], 1)
				sin_list_0 = array_ops.concat([sin_theta_0, -sin_theta_0], 1)			


			params_theta_1 = vs.get_variable("theta_1", [self._capacity/2, hidden_size/2-1], initializer=theta_phi_initializer)
			cos_theta_1 = math_ops.cos(params_theta_1)
			sin_theta_1 = math_ops.sin(params_theta_1)
			if self._comp:

				params_phi_1 = vs.get_variable("phi_1", [self._capacity/2, hidden_size/2-1], initializer=theta_phi_initializer)
				cos_phi_1 = math_ops.cos(params_phi_1)
				sin_phi_1 = math_ops.sin(params_phi_1)

				cos_list_1_re = array_ops.concat([np.ones((capacity/2,1)), cos_theta_1, math_ops.multiply(cos_theta_1, cos_phi_1), np.ones((capacity/2,1))], 1)
				cos_list_1_im = array_ops.concat([np.zeros((capacity/2,1)), array_ops.zeros_like(cos_theta_1), math_ops.multiply(cos_theta_1, sin_phi_1), np.zeros((capacity/2,1))], 1)
				sin_list_1_re = array_ops.concat([np.zeros((capacity/2,1)), sin_theta_1, -math_ops.multiply(sin_theta_1, cos_phi_1), np.zeros((capacity/2,1))], 1)
				sin_list_1_im = array_ops.concat([np.zeros((capacity/2,1)), array_ops.zeros_like(sin_theta_1), -math_ops.multiply(sin_theta_1, sin_phi_1), np.zeros((capacity/2,1))], 1)
				cos_list_1 = array_ops.unstack(math_ops.complex(cos_list_1_re, cos_list_1_im))
				sin_list_1 = array_ops.unstack(math_ops.complex(sin_list_1_re, sin_list_1_im))

			else:
				cos_list_1 = array_ops.concat([np.ones((capacity/2,1)), cos_theta_1, cos_theta_1, np.ones((capacity/2,1))], 1)
				sin_list_1 = array_ops.concat([np.zeros((capacity/2,1)), sin_theta_1, -sin_theta_1, np.zeros((capacity/2,1))], 1)
	





			ind, ind3, ind4 = permute_tunable(hidden_size, capacity)
			
			diag_list_0 = permute(cos_list_0, ind3)
			off_list_0 = permute(sin_list_0, ind3)
			diag_list_1 = permute(cos_list_1, ind4)
			off_list_1 = permute(sin_list_1, ind4)

			v1 = tf.reshape(tf.concat([diag_list_0, diag_list_1], 1), [capacity, hidden_size])
			v2 = tf.reshape(tf.concat([off_list_0, off_list_1], 1), [capacity, hidden_size])


		if self._comp:
			omega = vs.get_variable("omega", [hidden_size], initializer=theta_phi_initializer)
			D = math_ops.complex(math_ops.cos(omega), math_ops.sin(omega))
		else:
			D = None

		self.v1 = toTensorArray(v1)
		self.v2 = toTensorArray(v2)
		self.ind = toTensorArray(ind)
		self.diag = D



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

			Wh = EUNN(state, self._capacity, self.v1, self.v2, self.ind, self.diag)

			if self._comp:
				U_re = vs.get_variable("U_re", [inputs.get_shape()[-1], self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
				U_im = vs.get_variable("U_im", [inputs.get_shape()[-1], self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
				Ux_re = math_ops.matmul(inputs, U_re)
				Ux_im = math_ops.matmul(inputs, U_im)
				Ux = math_ops.complex(Ux_re, Ux_im)
			else:
				U = vs.get_variable("U", [inputs.get_shape()[-1], self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
				Ux = math_ops.matmul(inputs, U) 

			bias = vs.get_variable("modReLUBias", [self._hidden_size], initializer= init_ops.constant_initializer())
			output = self._activation((Ux + Wh), bias, self._comp)  

		return output, output

