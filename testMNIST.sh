echo ---------EURNN Ivan---------------
python2.7 mnist_task.py EURNNIvan -I $1 -F False -C False -B 128 -H 128 -L 8 

echo ---------EURNN Orig---------------
python2.7 mnist_task.py EURNN -I $1 -F False -C False -B 128 -H 128 -L 8
