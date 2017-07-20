echo ---------EURNN Ivan---------------
python2.7 mnist_task.py EURNNIvan -I $1 -F True -C False -B 128 -H 128 

echo ---------EURNN Orig---------------
python2.7 mnist_task.py EURNN -I $1 -F True -C False -B 128 -H 128
