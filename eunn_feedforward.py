import tensorflow as tf
import math
import numpy as np
from eunn import *


def eunn_feedforward(x):
    """
    feedforward layer using eunn 
    input: x of shape [None, 2^N]
    output: same shape as x

    Note: 
    1. use different scope if this layer is called multiple times
    2. only support fft style in the paper
    3. only support real number, so it is actually orthogonal matrix

    """
    d = int(x.shape[-1])
    capacity = int(math.log(d, 2))
    v1, v2, ind, _ = fft_param(d, False)
    h = x
    for i in range(capacity):
        diag = h * v1[i]
        off = h * v2[i]
        h = diag + tf.gather(off, ind[i], axis=1)
    return h

# 
# if __name__ == '__main__':
#     x = tf.placeholder(tf.float32, [None, 1024])
#     y = eunn_feedforward(x)
#     print(y)




