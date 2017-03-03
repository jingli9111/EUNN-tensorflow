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
	ind1 = range(s)
	ind2 = range(s)

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

	ind = [ind1, ind2] * (L/2)

	ind3 = []
	ind4 = []

	for i in range(s/2):
		ind3.append(i)
		ind3.append(i + s/2)

	ind4.append(0)
	for i in range(s/2 - 1):
		ind4.append(i + 1)
		ind4.append(i + s/2)
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





def EUNN(h, L, v1_list, v2_list, ind_list, D):

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



