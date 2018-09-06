# EUNN-tensorflow

Unitary neural network is able to solve gradient vanishing and gradient explosion problem and help learning long term dependency. EUNN is an efficient and strictly enforced unitary parametrization based on SU(2) group. This repository contains the implementation of Efficient Unitary Neural Network(EUNN) in tensorflow. 

If you find this work useful, please cite [arXiv:1612.05231](https://arxiv.org/pdf/1612.05231.pdf). 

I am working on submitting this code to `tf.contrib` so that in the future you can use it directly from official tensorflow.

## Installation

requires TensorFlow > 1.2.0

## Demo

```
./demo.sh
```

## Usage

To use EUNN in your model, simply copy [eunn.py](https://github.com/jingli9111/EUNN-tensorflow/blob/master/eunn.py).

Then you can use EUNN in the same way you use built-in LSTM:
```
from eunn import EUNNCell
cell = EUNNCell(hidden_size, capacity, fft, complex)
```
Args:
- `hidden_size`: `Integer`.
- `capacity`: `Optional`. `Integer`. Only works for tunable style.
- `fft`: `Optional`. `Bool`. If `True`, EUNN is set to FFT style. Default is `False`.
- `complex`: `Optional`. `Bool`. If `True`, EUNN is set to complex domain. Default is `True`.

Note:
- For complex domain, the data type should be `tf.complex64`
- For real domain, the data type should be `tf.float32`


## Example tasks for EUNN
Copying memory task and pixel-permuted MNIST task for RNN in the paper are shown here. 
Due to copyright issue, we cannot release TIMIT task.

#### Copying Memory Task
```
python copying_task.py --model eunn --T 200 --fft
```


#### Pixel-Permuted MNIST Task
```
python mnist_task.py --model eunn --iter 20000 --hidden 512 --complex False 
```

####

## Licese 
MIT License

