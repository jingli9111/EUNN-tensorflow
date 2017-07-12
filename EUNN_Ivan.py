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
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs

def toTensorArray(elems):
    
    elems = ops.convert_to_tensor(elems)
    n = array_ops.shape(elems)[0]
    elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, size=n, dynamic_size=False, infer_shape=True, clear_after_read = False)
    elems_ta = elems_ta.unstack(elems)
    return elems_ta


def EUNN_param(hidden_size, capacity=2, FFT=False, comp=False):
    
    theta_phi_initializer = init_ops.random_uniform_initializer(-np.pi, np.pi)
    
    sizeB = capacity//2
    sizeA = capacity - sizeB

    params_theta_0 = vs.get_variable("theta_0", [sizeA, int(hidden_size/2)], initializer=theta_phi_initializer)
    cos_theta_0 = array_ops.reshape(math_ops.cos(params_theta_0),[sizeA,-1,1])
    sin_theta_0 = array_ops.reshape(math_ops.sin(params_theta_0),[sizeA,-1,1])
 
    params_theta_1 = vs.get_variable("theta_1", [sizeB, int(hidden_size/2)-1], initializer=theta_phi_initializer)
    cos_theta_1 = array_ops.reshape(math_ops.cos(params_theta_1),[sizeB,-1,1])
    sin_theta_1 = array_ops.reshape(math_ops.sin(params_theta_1),[sizeB,-1,1])

    if comp:
        params_phi_0 = vs.get_variable("phi_0", [sizeA, int(hidden_size/2)], initializer=theta_phi_initializer)
        cos_phi_0 = array_ops.reshape(math_ops.cos(params_phi_0),[sizeA,-1,1])
        sin_phi_0 = array_ops.reshape(math_ops.sin(params_phi_0),[sizeA,-1,1])

        cos_list_0_re = array_ops.reshape(array_ops.concat([cos_theta_0, math_ops.multiply(cos_theta_0, cos_phi_0)], 2),[sizeA,-1])
        cos_list_0_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(cos_theta_0), math_ops.multiply(cos_theta_0, sin_phi_0)], 2),[sizeA,-1])
        cos_list_0 = math_ops.complex(cos_list_0_re, cos_list_0_im)
        
        sin_list_0_re = array_ops.reshape(array_ops.concat([sin_theta_0, -math_ops.multiply(sin_theta_0, cos_phi_0)], 2),[sizeA,-1])
        sin_list_0_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(sin_theta_0), -math_ops.multiply(sin_theta_0, sin_phi_0)], 2),[sizeA,-1])
        sin_list_0 = math_ops.complex(sin_list_0_re, sin_list_0_im)

        params_phi_1 = vs.get_variable("phi_1", [sizeB, int(hidden_size/2)-1], initializer=theta_phi_initializer)
        cos_phi_1 = array_ops.reshape(math_ops.cos(params_phi_1),[sizeB,-1,1])
        sin_phi_1 = array_ops.reshape(math_ops.sin(params_phi_1),[sizeB,-1,1])

        cos_list_1_re = array_ops.reshape(array_ops.concat([cos_theta_1, math_ops.multiply(cos_theta_1, cos_phi_1)], 2),[sizeB,-1])
        cos_list_1_re = array_ops.concat([tf.ones((sizeB,1)), cos_list_1_re, tf.ones((sizeB,1))], 1)
        cos_list_1_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(cos_theta_1), math_ops.multiply(cos_theta_1, sin_phi_1)], 2),[sizeB,-1])
        cos_list_1_im = array_ops.concat([tf.zeros((sizeB,1)), cos_list_1_im, tf.zeros((sizeB,1))], 1)
        cos_list_1 = math_ops.complex(cos_list_1_re, cos_list_1_im)
        
        sin_list_1_re = array_ops.reshape(array_ops.concat([sin_theta_1, -math_ops.multiply(sin_theta_1, cos_phi_1)], 2),[sizeB,-1])
        sin_list_1_re = array_ops.concat([tf.zeros((sizeB,1)), sin_list_1_re, tf.zeros((sizeB,1))], 1)
        sin_list_1_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(sin_theta_1), -math_ops.multiply(sin_theta_1, sin_phi_1)], 2),[sizeB,-1])
        sin_list_1_im = array_ops.concat([tf.zeros((sizeB,1)), sin_list_1_im, tf.zeros((sizeB,1))], 1)
        sin_list_1 = math_ops.complex(sin_list_1_re, sin_list_1_im)
    else:
        cos_list_0 = array_ops.reshape(array_ops.concat([cos_theta_0, cos_theta_0], 2),[sizeA,-1])
        sin_list_0 = array_ops.reshape(array_ops.concat([sin_theta_0, -sin_theta_0], 2),[sizeA,-1])
        
        cos_list_1 = array_ops.reshape(array_ops.concat([cos_theta_1, cos_theta_1],2),[sizeB,-1])
        cos_list_1 = array_ops.concat([tf.ones((sizeB,1)),cos_list_1,tf.ones((sizeB,1))],1)
        
        sin_list_1 = array_ops.reshape(array_ops.concat([sin_theta_1, -sin_theta_1],2),[sizeB,-1])
        sin_list_1 = array_ops.concat([tf.zeros((sizeB,1)),sin_list_1,tf.zeros((sizeB,1))],1);
    
    if sizeB != sizeA:
        if comp:
            cos_list_1 = array_ops.concat([cos_list_1,math_ops.complex(tf.zeros([1,hidden_size]),tf.zeros([1,hidden_size]))],0)
            sin_list_1 = array_ops.concat([sin_list_1,math_ops.complex(tf.zeros([1,hidden_size]),tf.zeros([1,hidden_size]))],0)
        else:
            cos_list_1 = array_ops.concat([cos_list_1,tf.zeros([1,hidden_size])],0)
            sin_list_1 = array_ops.concat([sin_list_1,tf.zeros([1,hidden_size])],0)

    v1 = tf.reshape(tf.concat([cos_list_0, cos_list_1], 1), [sizeA*2, hidden_size])
    v2 = tf.reshape(tf.concat([sin_list_0, sin_list_1], 1), [sizeA*2, hidden_size])

    if sizeB != sizeA:
        v1 = tf.slice(v1,[0,0],[capacity,hidden_size])
        v2 = tf.slice(v2,[0,0],[capacity,hidden_size])

    v1 = toTensorArray(v1)
    v2 = toTensorArray(v2)
    if comp:
        omega = vs.get_variable("omega", [hidden_size], initializer=theta_phi_initializer)
        diag = math_ops.complex(math_ops.cos(omega), math_ops.sin(omega))
    else:
        diag = None
    
    return v1, v2, diag, capacity


def EUNN_loop(h, L, v1_list, v2_list, D):
   
    i = 0
    def F(x, i):

        v1 = v1_list.read(i)
        v2 = v2_list.read(i)
        
        diag = math_ops.multiply(x, v1)
        off = math_ops.multiply(x, v2)
     
        def evenI(off):
            s = int(off.get_shape()[1])
            off = array_ops.reshape(off,[-1,s//2,2])
            off = array_ops.reshape(array_ops.reverse(off,[2]),[-1,s])
            return off

        def oddI(off):
            s = int(off.get_shape()[1]) - 2
            helper1, off, helper2 = array_ops.split(off,[1,s,1],1)
            off = array_ops.reshape(off,[-1,s//2,2])
            off = array_ops.reshape(array_ops.reverse(off,[2]),[-1,s])
            off = array_ops.concat([helper1, off, helper2],1)
            return off

        off = control_flow_ops.cond(gen_math_ops.equal(gen_math_ops.mod(i,2),0), lambda: evenI(off), lambda: oddI(off))

        Fx = diag + off                                      
        i += 1                                                
                                                               
        return Fx, i                                          
                                                                   
    FFx, _ =  control_flow_ops.while_loop(lambda x, i: gen_math_ops.less(i, L), F, [h, i])                                                              
    if not D  == None:                                             
         Wx = math_ops.multiply(FFx, D)                        
    else:                                                          
         Wx = FFx                                              
                                                                   
    return Wx                                                     
