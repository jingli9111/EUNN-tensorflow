from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops

def modrelu(z, b, comp):
    if comp:
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