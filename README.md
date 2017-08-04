# EUNN-tensorflow

Unitary neural network is able to solve gradient vanishing and gradient explosion problem and help learning long term dependency. EUNN is an efficient and strictly enforced unitary parametrization based on SU(2) group. This repository contains the implementation of Efficient Unitary Neural Network(EUNN) in tensorflow. 

If you find this work useful, please cite [arXiv:1612.05231](https://arxiv.org/pdf/1612.05231.pdf). The current implementation is developed by [Ivan Ivanov](https://github.com/vanjo9800).

## Installation

requires TensorFlow 1.2.0

## Demo

```
./demo.sh
```

## Usage

#### Use EUNN in RNN 
To use EUNN in your model, simply copy [EUNN.py](https://github.com/jingli9111/EUNN-tensorflow/blob/master/EUNN.py).
You don't need to install or write C code.

Then you can use EUNN in the same way you use built-in LSTM:
```
from EUNN import EUNNCell
cell = EUNNCell(n_hidden, capacity=2, FFT=False, comp=True)
```
Args:
- `n_hidden`: `Integer`.
- `capacity`: `Optional`. `Integer`. Only works for tunable style.
- `FFT`: `Optional`. `Bool`. If `True`, EUNN is set to FFT style. Default is `False
- `comp`: `Optional`. `Bool`. If `True`, EUNN is set to complex domain. Default is `True`.

Note:
- For complex domain, the data type should be `tf.complex64`
- For real domain, the data type should be `tf.float32`


## Example tasks for EUNN
Two tasks for RNN in the paper are shown here. Use `-h` for more information

#### Copying Memory Task
```
python copying_task.py EUNN -T 200
```


#### Pixel-Permuted MNIST Task
```
python mnist_task.py EUNN -I 20000 -H 512 -C False 
```

