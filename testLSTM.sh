echo ---------LSTM Copying---------------
python2.7 copying_task.py LSTM -T 500 -I $1 -B 128 -H 128

echo ---------LSTM MNIST---------------
python2.7 mnist_task.py LSTM -I $1 -B 128 -H 128
