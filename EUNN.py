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
    if FFT:
        capacity = int(np.ceil(np.log2(hidden_size)))

        diag_list_0 = []
        off_list_0 = []
        varsize = 0
        for i in range(capacity):
            size = capacity - i
            normalSize = (hidden_size // (2 ** size )) * (2 ** (size - 1))
            extraSize = max(0, (hidden_size % (2 ** size)) - (2 ** (size - 1)))
            varsize += normalSize + extraSize

        params_theta = vs.get_variable("theta_0", [varsize], initializer=theta_phi_initializer)
        cos_theta = math_ops.cos(params_theta)
        sin_theta = math_ops.sin(params_theta)

        if comp:
            params_phi = vs.get_variable("phi_0", [varsize], initializer=theta_phi_initializer)
            cos_phi = math_ops.cos(params_phi)
            sin_phi = math_ops.sin(params_phi)

            cos_list_0 = math_ops.complex(cos_theta, array_ops.zeros_like(cos_theta))
            cos_list_1 = math_ops.complex(math_ops.multiply(cos_theta,cos_phi), math_ops.multiply(cos_theta,sin_phi))
            sin_list_0 = math_ops.complex(sin_theta, array_ops.zeros_like(sin_theta))
            sin_list_1 = math_ops.complex(-math_ops.multiply(sin_theta,cos_phi), -math_ops.multiply(sin_theta,sin_phi))

        last = 0
        for i in range(capacity):
            size = capacity - i
            normalSize = (hidden_size // (2 ** size )) * (2 ** (size - 1))
            extraSize = max(0, (hidden_size % (2 ** size)) - (2 ** (size - 1)))
            
            if comp:
                cos_list_normal = array_ops.concat([array_ops.slice(cos_list_0,[last],[normalSize]),array_ops.slice(cos_list_1,[last],[normalSize])],0)
                sin_list_normal = array_ops.concat([array_ops.slice(sin_list_0,[last],[normalSize]),array_ops.slice(sin_list_1,[last],[normalSize])],0)
                last += normalSize

                cos_list_extra = array_ops.concat([array_ops.slice(cos_list_0,[last],[extraSize]),math_ops.complex(tf.ones([hidden_size - 2*normalSize - 2*extraSize]), tf.zeros([hidden_size - 2*normalSize - 2*extraSize])),array_ops.slice(cos_list_1,[last],[extraSize])],0)
                sin_list_extra = array_ops.concat([array_ops.slice(sin_list_0,[last],[extraSize]),math_ops.complex(tf.zeros([hidden_size - 2*normalSize - 2*extraSize]), tf.zeros([hidden_size - 2*normalSize - 2*extraSize])),array_ops.slice(sin_list_1,[last],[extraSize])],0)
                last += extraSize
            
            else:
                cos_list_normal = array_ops.slice(cos_theta,[last],[normalSize])
                cos_list_normal = array_ops.concat([cos_list_normal, cos_list_normal], 0)
                cos_list_extra = array_ops.slice(cos_theta,[last+normalSize],[extraSize])
                cos_list_extra = array_ops.concat([cos_list_extra, tf.ones([hidden_size - 2*normalSize - 2*extraSize]), cos_list_extra], 0)
                
                sin_list_normal = array_ops.slice(sin_theta,[last],[normalSize])
                sin_list_normal = array_ops.concat([sin_list_normal, -sin_list_normal], 0)
                sin_list_extra = array_ops.slice(sin_theta,[last+normalSize],[extraSize])
                sin_list_extra = array_ops.concat([sin_list_extra, tf.zeros([hidden_size - 2*normalSize - 2*extraSize]), -sin_list_extra], 0)

                last += normalSize + extraSize

            if normalSize != 0:
                cos_list_normal = array_ops.reshape(array_ops.transpose(array_ops.reshape(cos_list_normal, [-1,2*normalSize//(2**size)])), [-1])
                sin_list_normal = array_ops.reshape(array_ops.transpose(array_ops.reshape(sin_list_normal, [-1,2*normalSize//(2**size)])), [-1])
            
            cos_list = array_ops.concat([cos_list_normal, cos_list_extra], 0)
            sin_list = array_ops.concat([sin_list_normal, sin_list_extra], 0)
            diag_list_0.append(cos_list)
            off_list_0.append(sin_list)

        v1 = array_ops.stack(diag_list_0, 0)
        v2 = array_ops.stack(off_list_0, 0)

    else:
        capacityB = capacity//2
        capacityA = capacity - capacityB

        hidden_sizeA = hidden_size//2
        hidden_sizeB = (hidden_size-1)//2
        
        params_theta_0 = vs.get_variable("theta_0", [capacityA, hidden_sizeA], initializer=theta_phi_initializer)
        cos_theta_0 = array_ops.reshape(math_ops.cos(params_theta_0),[capacityA,-1,1])
        sin_theta_0 = array_ops.reshape(math_ops.sin(params_theta_0),[capacityA,-1,1])
 
        params_theta_1 = vs.get_variable("theta_1", [capacityB, hidden_sizeB], initializer=theta_phi_initializer)
        cos_theta_1 = array_ops.reshape(math_ops.cos(params_theta_1),[capacityB,-1,1])
        sin_theta_1 = array_ops.reshape(math_ops.sin(params_theta_1),[capacityB,-1,1])

        if comp:
            params_phi_0 = vs.get_variable("phi_0", [capacityA, hidden_sizeA], initializer=theta_phi_initializer)
            cos_phi_0 = array_ops.reshape(math_ops.cos(params_phi_0),[capacityA,-1,1])
            sin_phi_0 = array_ops.reshape(math_ops.sin(params_phi_0),[capacityA,-1,1])

            cos_list_0_re = array_ops.reshape(array_ops.concat([cos_theta_0, math_ops.multiply(cos_theta_0, cos_phi_0)], 2),[capacityA,-1])
            cos_list_0_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(cos_theta_0), math_ops.multiply(cos_theta_0, sin_phi_0)], 2),[capacityA,-1])
            if hidden_sizeA*2 != hidden_size:
                cos_list_0_re = array_ops.concat([cos_list_0_re,tf.ones([capacityA,1])],1)
                cos_list_0_im = array_ops.concat([cos_list_0_im,tf.zeros([capacityA,1])],1)
            cos_list_0 = math_ops.complex(cos_list_0_re, cos_list_0_im)
            
            sin_list_0_re = array_ops.reshape(array_ops.concat([sin_theta_0, -math_ops.multiply(sin_theta_0, cos_phi_0)], 2),[capacityA,-1])
            sin_list_0_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(sin_theta_0), -math_ops.multiply(sin_theta_0, sin_phi_0)], 2),[capacityA,-1])
            if hidden_sizeA*2 != hidden_size:
                sin_list_0_re = array_ops.concat([sin_list_0_re,tf.zeros([capacityA,1])],1)
                sin_list_0_im = array_ops.concat([sin_list_0_im,tf.zeros([capacityA,1])],1)
            sin_list_0 = math_ops.complex(sin_list_0_re, sin_list_0_im)

            params_phi_1 = vs.get_variable("phi_1", [capacityB, hidden_sizeB], initializer=theta_phi_initializer)
            cos_phi_1 = array_ops.reshape(math_ops.cos(params_phi_1),[capacityB,-1,1])
            sin_phi_1 = array_ops.reshape(math_ops.sin(params_phi_1),[capacityB,-1,1])

            cos_list_1_re = array_ops.reshape(array_ops.concat([cos_theta_1, math_ops.multiply(cos_theta_1, cos_phi_1)], 2),[capacityB,-1])
            cos_list_1_re = array_ops.concat([tf.ones((capacityB,1)), cos_list_1_re], 1)
            cos_list_1_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(cos_theta_1), math_ops.multiply(cos_theta_1, sin_phi_1)], 2),[capacityB,-1])
            cos_list_1_im = array_ops.concat([tf.zeros((capacityB,1)), cos_list_1_im], 1)
            if hidden_sizeB*2 != hidden_size-1:
                cos_list_1_re = array_ops.concat([cos_list_1_re,tf.ones([capacityB,1])],1)
                cos_list_1_im = array_ops.concat([cos_list_1_im,tf.zeros([capacityB,1])],1)
            cos_list_1 = math_ops.complex(cos_list_1_re, cos_list_1_im)
            
            sin_list_1_re = array_ops.reshape(array_ops.concat([sin_theta_1, -math_ops.multiply(sin_theta_1, cos_phi_1)], 2),[capacityB,-1])
            sin_list_1_re = array_ops.concat([tf.zeros((capacityB,1)), sin_list_1_re], 1)
            sin_list_1_im = array_ops.reshape(array_ops.concat([array_ops.zeros_like(sin_theta_1), -math_ops.multiply(sin_theta_1, sin_phi_1)], 2),[capacityB,-1])
            sin_list_1_im = array_ops.concat([tf.zeros((capacityB,1)), sin_list_1_im], 1)
            if hidden_sizeB*2 != hidden_size-1:
                sin_list_1_re = array_ops.concat([sin_list_1_re,tf.zeros([capacityB,1])],1)
                sin_list_1_im = array_ops.concat([sin_list_1_im,tf.zeros([capacityB,1])],1)
            sin_list_1 = math_ops.complex(sin_list_1_re, sin_list_1_im)
        else:
            cos_list_0 = array_ops.reshape(array_ops.concat([cos_theta_0, cos_theta_0], 2),[capacityA,-1])
            sin_list_0 = array_ops.reshape(array_ops.concat([sin_theta_0, -sin_theta_0], 2),[capacityA,-1])
            if hidden_sizeA*2 != hidden_size:
                cos_list_0 = array_ops.concat([cos_list_0,tf.ones([capacityA,1])],1)
                sin_list_0 = array_ops.concat([sin_list_0,tf.zeros([capacityA,1])],1)
            
            cos_list_1 = array_ops.reshape(array_ops.concat([cos_theta_1, cos_theta_1],2),[capacityB,-1])
            cos_list_1 = array_ops.concat([tf.ones((capacityB,1)),cos_list_1],1)
            sin_list_1 = array_ops.reshape(array_ops.concat([sin_theta_1, -sin_theta_1],2),[capacityB,-1])
            sin_list_1 = array_ops.concat([tf.zeros((capacityB,1)),sin_list_1],1)
            if hidden_sizeB*2 != hidden_size-1:
                cos_list_1 = array_ops.concat([cos_list_1,tf.zeros([capacityB,1])],1)
                sin_list_1 = array_ops.concat([sin_list_1,tf.zeros([capacityB,1])],1)
        
        if capacityB != capacityA:
            if comp:
                cos_list_1 = array_ops.concat([cos_list_1,math_ops.complex(tf.zeros([1,hidden_size]),tf.zeros([1,hidden_size]))],0)
                sin_list_1 = array_ops.concat([sin_list_1,math_ops.complex(tf.zeros([1,hidden_size]),tf.zeros([1,hidden_size]))],0)
            else:
                cos_list_1 = array_ops.concat([cos_list_1,tf.zeros([1,hidden_size])],0)
                sin_list_1 = array_ops.concat([sin_list_1,tf.zeros([1,hidden_size])],0)

        v1 = tf.reshape(tf.concat([cos_list_0, cos_list_1], 1), [capacityA*2, hidden_size])
        v2 = tf.reshape(tf.concat([sin_list_0, sin_list_1], 1), [capacityA*2, hidden_size])

        if capacityB != capacityA:
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


def EUNN_loop(h, L, v1_list, v2_list, D, FFT):
   
    i = 0
    def F_tunable(x, i):

        v1 = v1_list.read(i)
        v2 = v2_list.read(i)
        
        diag = math_ops.multiply(x, v1)
        off = math_ops.multiply(x, v2)
                
        def evenI(off,s):

            def evenS(off,s):
                off = array_ops.reshape(off,[-1,s//2,2])
                off = array_ops.reshape(array_ops.reverse(off,[2]),[-1,s])
                return off

            def oddS(off,s):
                off, helper = array_ops.split(off,[s-1,1],1)
                s-=1
                off = evenS(off,s)
                off = array_ops.concat([off,helper],1)
                return off

            off = control_flow_ops.cond(gen_math_ops.equal(gen_math_ops.mod(s,2),0), lambda: evenS(off,s), lambda: oddS(off,s))
            return off

        def oddI(off,s):
            helper, off = array_ops.split(off,[1,s-1],1)
            s-=1
            off = evenI(off,s)
            off = array_ops.concat([helper, off],1)
            return off

        s = int(off.get_shape()[1])
        off = control_flow_ops.cond(gen_math_ops.equal(gen_math_ops.mod(i,2),0), lambda: evenI(off,s), lambda: oddI(off,s))

        Fx = diag + off                                      
        i += 1                                                
                                                               
        return Fx, i
    
    def F_FFT(x, i):

        v1 = v1_list.read(i)
        v2 = v2_list.read(i)
        diag = math_ops.multiply(x, v1)
        off = math_ops.multiply(x, v2)
               
        hidden_size = int(off.get_shape()[1])
        size = 2**i
        dist = L - i
        normalSize = (hidden_size // (2**dist)) * (2**(dist-1))
        normalSize *= 2
        extraSize = tf.maximum(0, (hidden_size % (2**dist)) - (2**(dist-1)))
        hidden_size -= normalSize

        def modify(off_normal,dist,normalSize):
            off_normal = array_ops.reshape(array_ops.reverse(array_ops.reshape(off_normal,[-1,normalSize//(2**dist),2,(2**(dist-1))]),[2]),[-1,normalSize])
            return off_normal

        def doNothing(off_normal):
            return off_normal

        off_normal, off_extra = array_ops.split(off,[normalSize,hidden_size],1)
        off_normal = control_flow_ops.cond(gen_math_ops.equal(normalSize//(2*size),0), lambda: doNothing(off_normal), lambda: modify(off_normal,dist,normalSize))
        helper1, helper2 = array_ops.split(off_extra,[hidden_size-extraSize,extraSize],1)
        off_extra = array_ops.concat([helper2,helper1],1)
        off = array_ops.concat([off_normal,off_extra],1)

        Fx = diag + off                                      
        i += 1                                                
                                                               
        return Fx, i                                           
    
    if FFT:
        F = F_FFT
    else:
        F = F_tunable
    FFx, _ =  control_flow_ops.while_loop(lambda x, i: gen_math_ops.less(i, L), F, [h, i])                                                              
    if not D  == None:                                             
         Wx = math_ops.multiply(FFx, D)                        
    else:                                                          
         Wx = FFx                                              
                                                                   
    return Wx                                                     
