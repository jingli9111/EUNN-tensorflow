"""Module implementing EUNN Cell.
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from modrelu import modrelu

def _eunn_param(hidden_size, capacity=2, fft=False, comp=True):
    """
    Create parameters and do the initial preparations
    """
    theta_phi_initializer = init_ops.random_uniform_initializer(-np.pi, np.pi)
    if fft:
        capacity = int(np.ceil(np.log2(hidden_size)))

        diag_list_0 = []
        off_list_0 = []
        varsize = 0
        for i in range(capacity):
            size = capacity - i
            normal_size = (hidden_size // (2 ** size)) * (2 ** (size - 1))
            extra_size = max(0, (hidden_size % (2 ** size)) - (2 ** (size - 1)))
            varsize += normal_size + extra_size

        params_theta = vs.get_variable("theta_0", [varsize], initializer=theta_phi_initializer)
        cos_theta = math_ops.cos(params_theta)
        sin_theta = math_ops.sin(params_theta)

        if comp:
            params_phi = vs.get_variable("phi_0", [varsize], initializer=theta_phi_initializer)
            cos_phi = math_ops.cos(params_phi)
            sin_phi = math_ops.sin(params_phi)

            cos_list_0 = math_ops.complex(cos_theta, array_ops.zeros_like(cos_theta))
            cos_list_1 = math_ops.complex(math_ops.multiply(cos_theta, cos_phi), math_ops.multiply(cos_theta, sin_phi))
            sin_list_0 = math_ops.complex(sin_theta, array_ops.zeros_like(sin_theta))
            sin_list_1 = math_ops.complex(-math_ops.multiply(sin_theta, cos_phi), -math_ops.multiply(sin_theta, sin_phi))

        last = 0
        for i in range(capacity):
            size = capacity - i
            normal_size = (hidden_size // (2 ** size)) * (2 ** (size - 1))
            extra_size = max(0, (hidden_size % (2 ** size)) - (2 ** (size - 1)))

            if comp:
                cos_list_normal = array_ops.concat([array_ops.slice(cos_list_0, [last], [normal_size]), array_ops.slice(cos_list_1, [last], [normal_size])], 0)
                sin_list_normal = array_ops.concat([array_ops.slice(sin_list_0, [last], [normal_size]), -array_ops.slice(sin_list_1, [last], [normal_size])], 0)
                last += normal_size

                cos_list_extra = array_ops.concat([array_ops.slice(cos_list_0, [last], [extra_size]), math_ops.complex(tf.ones([hidden_size - 2*normal_size - 2*extra_size]), tf.zeros([hidden_size - 2*normal_size - 2*extra_size])), array_ops.slice(cos_list_1, [last], [extra_size])], 0)
                sin_list_extra = array_ops.concat([array_ops.slice(sin_list_0, [last], [extra_size]), math_ops.complex(tf.zeros([hidden_size - 2*normal_size - 2*extra_size]), tf.zeros([hidden_size - 2*normal_size - 2*extra_size])), -array_ops.slice(sin_list_1, [last], [extra_size])], 0)
                last += extra_size

            else:
                cos_list_normal = array_ops.slice(cos_theta, [last], [normal_size])
                cos_list_normal = array_ops.concat([cos_list_normal, cos_list_normal], 0)
                cos_list_extra = array_ops.slice(cos_theta, [last+normal_size], [extra_size])
                cos_list_extra = array_ops.concat([cos_list_extra, tf.ones([hidden_size - 2*normal_size - 2*extra_size]), cos_list_extra], 0)

                sin_list_normal = array_ops.slice(sin_theta, [last], [normal_size])
                sin_list_normal = array_ops.concat([sin_list_normal, -sin_list_normal], 0)
                sin_list_extra = array_ops.slice(sin_theta, [last+normal_size], [extra_size])
                sin_list_extra = array_ops.concat([sin_list_extra, tf.zeros([hidden_size - 2*normal_size - 2*extra_size]), -sin_list_extra], 0)

                last += normal_size + extra_size

            if normal_size != 0:
                cos_list_normal = array_ops.reshape(array_ops.transpose(array_ops.reshape(cos_list_normal, [-1, 2*normal_size//(2**size)])), [-1])
                sin_list_normal = array_ops.reshape(array_ops.transpose(array_ops.reshape(sin_list_normal, [-1, 2*normal_size//(2**size)])), [-1])

            cos_list = array_ops.concat([cos_list_normal, cos_list_extra], 0)
            sin_list = array_ops.concat([sin_list_normal, sin_list_extra], 0)
            diag_list_0.append(cos_list)
            off_list_0.append(sin_list)

        diag_vec = array_ops.stack(diag_list_0, 0)
        off_vec = array_ops.stack(off_list_0, 0)

    else:
        capacity_b = capacity//2
        capacity_a = capacity - capacity_b

        hidden_size_a = hidden_size//2
        hidden_size_b = (hidden_size-1)//2

        params_theta_0 = vs.get_variable("theta_0", [capacity_a, hidden_size_a], initializer=theta_phi_initializer)
        cos_theta_0 = array_ops.reshape(math_ops.cos(params_theta_0), [capacity_a, -1, 1])
        sin_theta_0 = array_ops.reshape(math_ops.sin(params_theta_0), [capacity_a, -1, 1])

        params_theta_1 = vs.get_variable("theta_1", [capacity_b, hidden_size_b], initializer=theta_phi_initializer)
        cos_theta_1 = array_ops.reshape(math_ops.cos(params_theta_1), [capacity_b, -1, 1])
        sin_theta_1 = array_ops.reshape(math_ops.sin(params_theta_1), [capacity_b, -1, 1])

        if comp:
            params_phi_0 = vs.get_variable("phi_0", [capacity_a, hidden_size_a], initializer=theta_phi_initializer)
            cos_phi_0 = array_ops.reshape(math_ops.cos(params_phi_0), [capacity_a, -1, 1])
            sin_phi_0 = array_ops.reshape(math_ops.sin(params_phi_0), [capacity_a, -1, 1])

            cos_list_0_re = array_ops.reshape(array_ops.concat([cos_theta_0, math_ops.multiply(cos_theta_0, cos_phi_0)], 2), [capacity_a, -1])
            cos_list_0_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(cos_theta_0), math_ops.multiply(cos_theta_0, sin_phi_0)], 2), [capacity_a, -1])
            if hidden_size_a*2 != hidden_size:
                cos_list_0_re = array_ops.concat([cos_list_0_re, tf.ones([capacity_a, 1])], 1)
                cos_list_0_im = array_ops.concat([cos_list_0_im, tf.zeros([capacity_a, 1])], 1)
            cos_list_0 = math_ops.complex(cos_list_0_re, cos_list_0_im)

            sin_list_0_re = array_ops.reshape(array_ops.concat([sin_theta_0, - math_ops.multiply(sin_theta_0, cos_phi_0)], 2), [capacity_a, -1])
            sin_list_0_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(sin_theta_0), - math_ops.multiply(sin_theta_0, sin_phi_0)], 2), [capacity_a, -1])
            if hidden_size_a*2 != hidden_size:
                sin_list_0_re = array_ops.concat([sin_list_0_re, tf.zeros([capacity_a, 1])], 1)
                sin_list_0_im = array_ops.concat([sin_list_0_im, tf.zeros([capacity_a, 1])], 1)
            sin_list_0 = math_ops.complex(sin_list_0_re, sin_list_0_im)

            params_phi_1 = vs.get_variable("phi_1", [capacity_b, hidden_size_b], initializer=theta_phi_initializer)
            cos_phi_1 = array_ops.reshape(math_ops.cos(params_phi_1), [capacity_b, -1, 1])
            sin_phi_1 = array_ops.reshape(math_ops.sin(params_phi_1), [capacity_b, -1, 1])

            cos_list_1_re = array_ops.reshape(array_ops.concat([cos_theta_1, math_ops.multiply(cos_theta_1, cos_phi_1)], 2), [capacity_b, -1])
            cos_list_1_re = array_ops.concat([tf.ones((capacity_b, 1)), cos_list_1_re], 1)
            cos_list_1_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(cos_theta_1), math_ops.multiply(cos_theta_1, sin_phi_1)], 2), [capacity_b, -1])
            cos_list_1_im = array_ops.concat([tf.zeros((capacity_b, 1)), cos_list_1_im], 1)
            if hidden_size_b*2 != hidden_size-1:
                cos_list_1_re = array_ops.concat([cos_list_1_re, tf.ones([capacity_b, 1])], 1)
                cos_list_1_im = array_ops.concat([cos_list_1_im, tf.zeros([capacity_b, 1])], 1)
            cos_list_1 = math_ops.complex(cos_list_1_re, cos_list_1_im)

            sin_list_1_re = array_ops.reshape(array_ops.concat([sin_theta_1, -math_ops.multiply(sin_theta_1, cos_phi_1)], 2), [capacity_b, -1])
            sin_list_1_re = array_ops.concat([tf.zeros((capacity_b, 1)), sin_list_1_re], 1)
            sin_list_1_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(sin_theta_1), -math_ops.multiply(sin_theta_1, sin_phi_1)], 2), [capacity_b, -1])
            sin_list_1_im = array_ops.concat([tf.zeros((capacity_b, 1)), sin_list_1_im], 1)
            if hidden_size_b*2 != hidden_size-1:
                sin_list_1_re = array_ops.concat([sin_list_1_re, tf.zeros([capacity_b, 1])], 1)
                sin_list_1_im = array_ops.concat([sin_list_1_im, tf.zeros([capacity_b, 1])], 1)
            sin_list_1 = math_ops.complex(sin_list_1_re, sin_list_1_im)
        else:
            cos_list_0 = array_ops.reshape(array_ops.concat([cos_theta_0, cos_theta_0], 2), [capacity_a, -1])
            sin_list_0 = array_ops.reshape(array_ops.concat([sin_theta_0, -sin_theta_0], 2), [capacity_a, -1])
            if hidden_size_a*2 != hidden_size:
                cos_list_0 = array_ops.concat([cos_list_0, tf.ones([capacity_a, 1])], 1)
                sin_list_0 = array_ops.concat([sin_list_0, tf.zeros([capacity_a, 1])], 1)

            cos_list_1 = array_ops.reshape(array_ops.concat([cos_theta_1, cos_theta_1], 2), [capacity_b, -1])
            cos_list_1 = array_ops.concat([tf.ones((capacity_b, 1)), cos_list_1], 1)
            sin_list_1 = array_ops.reshape(array_ops.concat([sin_theta_1, -sin_theta_1], 2), [capacity_b, -1])
            sin_list_1 = array_ops.concat([tf.zeros((capacity_b, 1)), sin_list_1], 1)
            if hidden_size_b*2 != hidden_size-1:
                cos_list_1 = array_ops.concat([cos_list_1, tf.zeros([capacity_b, 1])], 1)
                sin_list_1 = array_ops.concat([sin_list_1, tf.zeros([capacity_b, 1])], 1)

        if capacity_b != capacity_a:
            if comp:
                cos_list_1 = array_ops.concat([cos_list_1, math_ops.complex(tf.zeros([1, hidden_size]), tf.zeros([1, hidden_size]))], 0)
                sin_list_1 = array_ops.concat([sin_list_1, math_ops.complex(tf.zeros([1, hidden_size]), tf.zeros([1, hidden_size]))], 0)
            else:
                cos_list_1 = array_ops.concat([cos_list_1, tf.zeros([1, hidden_size])], 0)
                sin_list_1 = array_ops.concat([sin_list_1, tf.zeros([1, hidden_size])], 0)

        diag_vec = tf.reshape(tf.concat([cos_list_0, cos_list_1], 1), [capacity_a*2, hidden_size])
        off_vec = tf.reshape(tf.concat([sin_list_0, sin_list_1], 1), [capacity_a*2, hidden_size])

        if capacity_b != capacity_a:
            diag_vec = tf.slice(diag_vec, [0, 0], [capacity, hidden_size])
            off_vec = tf.slice(off_vec, [0, 0], [capacity, hidden_size])

    def _toTensorArray(elems):

        elems = ops.convert_to_tensor(elems)
        n = array_ops.shape(elems)[0]
        elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, size=n, dynamic_size=False, infer_shape=True, clear_after_read=False)
        elems_ta = elems_ta.unstack(elems)
        return elems_ta

    diag_vec = _toTensorArray(diag_vec)
    off_vec = _toTensorArray(off_vec)
    if comp:
        omega = vs.get_variable("omega", [hidden_size], initializer=theta_phi_initializer)
        diag = math_ops.complex(math_ops.cos(omega), math_ops.sin(omega))
    else:
        diag = None

    return diag_vec, off_vec, diag, capacity


def _eunn_loop(state, capacity, diag_vec_list, off_vec_list, diag, fft):
    """
    EUNN main loop, applying unitary matrix on input tensor
    """
    i = 0
    def layer_tunable(x, i):

        diag_vec = diag_vec_list.read(i)
        off_vec = off_vec_list.read(i)

        diag = math_ops.multiply(x, diag_vec)
        off = math_ops.multiply(x, off_vec)

        def even_input(off, size):

            def even_s(off, size):
                off = array_ops.reshape(off, [-1, size//2, 2])
                off = array_ops.reshape(array_ops.reverse(off, [2]), [-1, size])
                return off

            def odd_s(off, size):
                off, helper = array_ops.split(off, [size-1, 1], 1)
                size -= 1
                off = even_s(off, size)
                off = array_ops.concat([off, helper], 1)
                return off

            off = control_flow_ops.cond(gen_math_ops.equal(gen_math_ops.mod(size, 2), 0), lambda: even_s(off, size), lambda: odd_s(off, size))
            return off

        def odd_input(off, size):
            helper, off = array_ops.split(off, [1, size-1], 1)
            size -= 1
            off = even_input(off, size)
            off = array_ops.concat([helper, off], 1)
            return off

        size = int(off.get_shape()[1])
        off = control_flow_ops.cond(gen_math_ops.equal(gen_math_ops.mod(i, 2), 0), lambda: even_input(off, size), lambda: odd_input(off, size))

        layer_output = diag + off
        i += 1

        return layer_output, i

    def layer_fft(state, i):

        diag_vec = diag_vec_list.read(i)
        off_vec = off_vec_list.read(i)
        diag = math_ops.multiply(state, diag_vec)
        off = math_ops.multiply(state, off_vec)

        hidden_size = int(off.get_shape()[1])
        # size = 2**i
        dist = capacity - i
        normal_size = (hidden_size // (2**dist)) * (2**(dist-1))
        normal_size *= 2
        extra_size = tf.maximum(0, (hidden_size % (2**dist)) - (2**(dist-1)))
        hidden_size -= normal_size

        def modify(off_normal, dist, normal_size):
            off_normal = array_ops.reshape(array_ops.reverse(array_ops.reshape(off_normal, [-1, normal_size//(2**dist), 2, (2**(dist-1))]), [2]), [-1, normal_size])
            return off_normal

        def do_nothing(off_normal):
            return off_normal

        off_normal, off_extra = array_ops.split(off, [normal_size, hidden_size], 1)
        off_normal = control_flow_ops.cond(gen_math_ops.equal(normal_size, 0), lambda: do_nothing(off_normal), lambda: modify(off_normal, dist, normal_size))
        helper1, helper2 = array_ops.split(off_extra, [hidden_size-extra_size, extra_size], 1)
        off_extra = array_ops.concat([helper2, helper1], 1)
        off = array_ops.concat([off_normal, off_extra], 1)

        layer_output = diag + off
        i += 1

        return layer_output, i

    if fft:
        layer_function = layer_fft
    else:
        layer_function = layer_tunable
    output, _ = control_flow_ops.while_loop(lambda state, i: gen_math_ops.less(i, capacity), layer_function, [state, i])

    if not diag is None:
        output = math_ops.multiply(output, diag)


    return output

class EUNNCell(RNNCell):
    """Efficient Unitary Network Cell
    The implementation is based on: http://arxiv.org/abs/1612.05231.

    """

    def __init__(self, hidden_size, capacity=2, fft=False, comp=False, activation=modrelu):
        super(EUNNCell, self).__init__()
        self._hidden_size = hidden_size
        self._activation = activation
        self._capacity = capacity
        self._fft = fft
        self._comp = comp

        self.diag_vec, self.off_vec, self.diag, self._capacity = _eunn_param(hidden_size, capacity, fft, comp)



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
        with vs.variable_scope(scope or "eunn_cell"):

            state = _eunn_loop(state, self._capacity, self.diag_vec, self.off_vec, self.diag, self._fft)

            input_matrix_init = init_ops.random_uniform_initializer(-0.01, 0.01)
            if self._comp:
                input_matrix_re = vs.get_variable("U_re", [inputs.get_shape()[-1], self._hidden_size], initializer=input_matrix_init)
                input_matrix_im = vs.get_variable("U_im", [inputs.get_shape()[-1], self._hidden_size], initializer=input_matrix_init)
                inputs_re = math_ops.matmul(inputs, input_matrix_re)
                inputs_im = math_ops.matmul(inputs, input_matrix_im)
                inputs = math_ops.complex(inputs_re, inputs_im)
            else:
                input_matrix = vs.get_variable("U", [inputs.get_shape()[-1], self._hidden_size], initializer=input_matrix_init)
                inputs = math_ops.matmul(inputs, input_matrix)

            bias = vs.get_variable("modReLUBias", [self._hidden_size], initializer=init_ops.constant_initializer())
            output = self._activation((inputs + state), bias, self._comp)

        return output, output
