# EUNN-tensorflow

Unitary neural network is able to solve gradient vanishing and gradient explosion problem and help learning long term correlation. Unitary RNN is promising to replace LSTM in multiple tasks. EUNN is an efficient unitary architecture based on SU(2) group. This repository contains an implementation of Efficient Unitary Neural Network(EUNN) and its Recurrent Neural Network implementation(EURNN). 

If you find this work useful, please cite [arXiv:1612.05231](https://arxiv.org/pdf/1612.05231.pdf).

## Installation

requires TensorFlow 1.0.0

## Demo

```
./demo.sh
```

## Usage

#### Use EUNN in RNN 
To use EURNN in your model, simply copy [EUNN.py](https://github.com/jingli9111/EUNN-tensorflow/blob/master/EUNN.py) and [EURNN.py](https://github.com/jingli9111/EUNN-tensorflow/blob/master/EURNN.py) files.

Then you can use EURNN in the same way you use built-in LSTM:
```
from EURNN import EURNNCell
cell = EURNNCell(n_hidden, capacity=2, FFT=False, comp=False)
```
Args:
- `n_hidden`: `Integer`. For FFT style, must be power of 2.
- `capacity`: `Optional`. `Integer`. Only works for tunable style, must be even number.
- `FFT`: `Optional`. `Bool`. If `True`, EURNN is set to FFT style. Default is `False`.
- `comp`: `Optional`. `Bool`. If `True`, EURNN is set to complex domain. Default is `False`.

Note:
- For complex domain, the data type should be `tf.complex64`
- For real domain, the data type should be `tf.float32`


#### Use EUNN in other applications
To use EUNN in your model, simply copy [EUNN.py](https://github.com/jingli9111/EUNN-tensorflow/blob/master/EUNN.py) file.

Then you can use EUNN in the following way:
```
from EUNN import EUNN
output = EUNN(input, capacity=2, FFT=False, comp=False)
```
Args:
- `input`: `2D-Tensor`. For FFT style, dimension must be power of 2.
- `capacity`: `Optional`. `Integer`. Only works for tunable style, must be even number.
- `FFT`: `Optional`. `Bool`. If `True`, EUNN is set to FFT style. Default is `False`.
- `comp`: `Optional`. `Bool`. If `True`, EUNN is set to complex domain. Default is `False`.

Note:
- For complex domain, the data type should be `tf.complex64`
- For real domain, the data type should be `tf.float32`



## Example tasks for EURNN
Two tasks for RNN in the paper are shown here. Use `-h` for more information

#### Copying Memory Task
requires: Model name (`EURNN` or `LSTM`);

optional parameters for the task: 

delay time`-T`, number of iterations`-I`, batch size`-B`, hidden size`-H`;

optional parameters for EURNN:

capacity`-L`, complex or real`-C`, FFT style or tunable style`-F`.

Example:
```
python copying_task.py EURNN -T 100 -I 2000 -B 128 -H 128 -C True -F True
```


#### Pixel-Permuted MNIST Task
requires: Model name (`EURNN` or `LSTM`);

optional parameters for the task:

number of iterations`-I`, batch size`-B`, hidden size`-H`;

optional parameters for EURNN:

capacity`-L`, complex or real`-C`, FFT style or tunable style`-F`.

Example:
```
python mnist_task.py EURNN -I 2000 -B 128 -H 128 -L 4 -C True 
```

