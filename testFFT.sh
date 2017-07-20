echo ---------EURNN Ivan---------------
python2.7 copying_task.py EURNNIvan -T 500 -I $1 -F True -C True -B 128 -H 128

echo ---------EURNN Orig---------------
python2.7 copying_task.py EURNN -T 500 -I $1 -F True -C True -B 128 -H 128
