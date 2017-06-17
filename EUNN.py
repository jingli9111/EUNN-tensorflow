import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs


def permute(x, ind):
	step1 = array_ops.transpose(x)
	step2 = array_ops.gather(step1, ind)
	step3 = array_ops.transpose(step2)
	return step3

def permute_tunable(s, L):
	ind1 = list(range(s))
	ind2 = list(range(s))

	for i in range(s):

		if i%2 == 1:
			ind1[i] = ind1[i] - 1
			if i == s -1:
				continue
			else:
				ind2[i] = ind2[i] + 1
		else:
			ind1[i] = ind1[i] + 1
			if i == 0: 
				continue
			else:
				ind2[i] = ind2[i] - 1

	ind = [ind1, ind2] * int(L/2)

	ind3 = []
	ind4 = []

	for i in range(int(s/2)):
		ind3.append(i)
		ind3.append(i + int(s/2))

	ind4.append(0)
	for i in range(int(s/2) - 1):
		ind4.append(i + 1)
		ind4.append(i + int(s/2))
	ind4.append(s - 1)

	return [ind, ind3, ind4]


def permute_FFT(s):
	def ind_s(k):
		if k==0:
			return np.array([[1,0]])
		else:
			temp = np.array(range(2**k))
			list0 = [np.append(temp + 2**k, temp)]
			list1 = ind_s(k-1)
			for i in range(k):
				list0.append(np.append(list1[i],list1[i] + 2**k))
			return list0

	t = ind_s(int(np.log2(s/2)))

	ind_list5 = []
	for i in range(int(np.log2(s))):
		ind_list5.append(tf.constant(t[i]))

	ind_list6 = []
	for i in range(int(np.log2(s))):
		ind = np.array([])
		for j in range(2**i):
			ind = np.append(ind, np.array(range(0, s, 2**i)) + j).astype(np.int32)

		ind_list6.append(tf.constant(ind))
	return ind_list5, ind_list6


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


def EUNN_param(hidden_size, capacity=2, FFT=False, comp=False):
	
	theta_phi_initializer = init_ops.random_uniform_initializer(-np.pi, np.pi)
	if FFT:
		capacity = int(np.log2(hidden_size))

		params_theta_0 = vs.get_variable("theta_0", [capacity, hidden_size/2], initializer=theta_phi_initializer)
		cos_theta_0 = math_ops.cos(params_theta_0)
		sin_theta_0 = math_ops.sin(params_theta_0)
		
		if comp:

			params_phi_0 = vs.get_variable("phi_0", [capacity, hidden_size/2], initializer=theta_phi_initializer)
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
		for i in range(capacity):
			diag_list_0.append(permute(cos_list_0[i], ind1_list[i]))
			off_list_0.append(permute(sin_list_0[i], ind1_list[i]))
		v1 = array_ops.stack(diag_list_0, 0)
		v2 = array_ops.stack(off_list_0, 0)

	else:

		params_theta_0 = vs.get_variable("theta_0", [int(capacity/2), int(hidden_size/2)], initializer=theta_phi_initializer)
		cos_theta_0 = math_ops.cos(params_theta_0)
		sin_theta_0 = math_ops.sin(params_theta_0)

		if comp:
			params_phi_0 = vs.get_variable("phi_0", [int(capacity/2), int(hidden_size/2)], initializer=theta_phi_initializer)
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


		params_theta_1 = vs.get_variable("theta_1", [int(capacity/2), int(hidden_size/2)-1], initializer=theta_phi_initializer)
		cos_theta_1 = math_ops.cos(params_theta_1)
		sin_theta_1 = math_ops.sin(params_theta_1)

		if comp:
			params_phi_1 = vs.get_variable("phi_1", [int(capacity/2), int(hidden_size/2)-1], initializer=theta_phi_initializer)
			cos_phi_1 = math_ops.cos(params_phi_1)
			sin_phi_1 = math_ops.sin(params_phi_1)

			cos_list_1_re = array_ops.concat([np.ones((int(capacity/2),1)), cos_theta_1, math_ops.multiply(cos_theta_1, cos_phi_1), np.ones((int(capacity/2),1))], 1)
			cos_list_1_im = array_ops.concat([np.zeros((int(capacity/2),1)), array_ops.zeros_like(cos_theta_1), math_ops.multiply(cos_theta_1, sin_phi_1), np.zeros((int(capacity/2),1))], 1)
			sin_list_1_re = array_ops.concat([np.zeros((int(capacity/2),1)), sin_theta_1, -math_ops.multiply(sin_theta_1, cos_phi_1), np.zeros((int(capacity/2),1))], 1)
			sin_list_1_im = array_ops.concat([np.zeros((int(capacity/2),1)), array_ops.zeros_like(sin_theta_1), -math_ops.multiply(sin_theta_1, sin_phi_1), np.zeros((int(capacity/2),1))], 1)
			cos_list_1 = array_ops.unstack(math_ops.complex(cos_list_1_re, cos_list_1_im))
			sin_list_1 = array_ops.unstack(math_ops.complex(sin_list_1_re, sin_list_1_im))
		else:
			cos_list_1 = array_ops.concat([np.ones((int(capacity/2),1)), cos_theta_1, cos_theta_1, np.ones((int(capacity/2),1))], 1)
			sin_list_1 = array_ops.concat([np.zeros((int(capacity/2),1)), sin_theta_1, -sin_theta_1, np.zeros((int(capacity/2),1))], 1)






		ind, ind3, ind4 = permute_tunable(hidden_size, capacity)
		
		diag_list_0 = permute(cos_list_0, ind3)
		off_list_0 = permute(sin_list_0, ind3)
		diag_list_1 = permute(cos_list_1, ind4)
		off_list_1 = permute(sin_list_1, ind4)

		v1 = tf.reshape(tf.concat([diag_list_0, diag_list_1], 1), [capacity, hidden_size])
		v2 = tf.reshape(tf.concat([off_list_0, off_list_1], 1), [capacity, hidden_size])


	if comp:
		omega = vs.get_variable("omega", [hidden_size], initializer=theta_phi_initializer)
		D = math_ops.complex(math_ops.cos(omega), math_ops.sin(omega))
	else:
		D = None

	v1 = toTensorArray(v1)
	v2 = toTensorArray(v2)
	ind = toTensorArray(ind)
	diag = D

	return v1, v2, ind, diag, capacity


def EUNN_loop(h, L, v1_list, v2_list, ind_list, D):

	i = 0

	def F(x, i):


		v1 = v1_list.read(i)
		v2 = v2_list.read(i)
		ind = ind_list.read(i)

		diag = math_ops.multiply(x, v1)
		off = math_ops.multiply(x, v2)
		Fx = diag + permute(off, ind)

		i += 1

		return Fx, i

	def cond(x, i):
		return i < L

	loop_vars = [h, i]
	FFx, _ = control_flow_ops.while_loop(
		cond, 
		F, 
		loop_vars
	)

	if not D == None:
		Wx = math_ops.multiply(FFx, D)
	else:
		Wx = FFx

	return Wx

def EUNN(input, capacity=2, FFT=False, comp=False):

	hidden_size = int(input.get_shape()[-1])
	v1, v2, ind, diag, capacity = EUNN_param(hidden_size, capacity, FFT, comp)

	output = EUNN_loop(input, capacity, v1, v2, ind, diag)

	return output
