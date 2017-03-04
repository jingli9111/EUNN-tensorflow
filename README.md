# EUNN-tensorflow

This repository contains an implementation of Efficient Unitary Neural Network(EUNN) and its Recurrent Neural Network implementation(EURNN). For more detail, see [arXiv:1612.05231](https://arxiv.org/pdf/1612.05231.pdf)

## Installation

requires TensorFlow 1.0

## Usage

#### Use EUNN in RNN 
To use EURNN in your model, simply copy [EUNN.py](https://github.com/jingli9111/EUNN-tensorflow/blob/master/EUNN.py) and [eurnn.py](https://github.com/jingli9111/EUNN-tensorflow/blob/master/eurnn.py) files.

Then you can use EURNN in the same way you use built-in LSTM:
```
from eurnn import EURNNCell
cell = EURNNCell(n_hidden, capacity, FFT, comp)
```
Args:
- n_hidden: An integer. For FFT style, must be power of 2.
- capacity: An integer. Only works for tunable style, must be even number.
- FFT: bool. Indicate using FFT style.
- comp: bool. Indicate using complex domain.

Note:
- For complex domain, the data type should be tf.complex64
- For real domain, the data type should be tf.float32


#### Use EUNN in other applications
Since we put the procedure to generate unitary matrices in EURNNCell.\_\_init\_\_() in order to avoiding redundant calculations, only [EUNN.py](https://github.com/jingli9111/EUNN-tensorflow/blob/master/EUNN.py) is not able to work as a unitary layer. We will develop this function soon.



## Example tasks
Two tasks for RNN in the paper are shown here. 

#### Copying Memory Task
Copying Memory Task requires: Model name;

optional parameters are: delay time, number of iterations, batch size, hidden size;

optional parameters for EURNN are: capacity, complex or real, FFT style or tunable style.

Example:
```
python copying_task.py EURNN -T 100 -I 2000 -B 128 -H 128 -C True -F True
```


#### Pixel-Permuted MNIST Task
Pixel-Permuted MNIST Task requires: Model name;

optional parameters are: number of iterations, batch size, hidden size;

optional parameters for EURNN are: capacity, complex or real, FFT style or tunable style.

Example:
```
python mnist_task.py EURNN -I 2000 -B 128 -H 128 -L 4 -C True 
```

