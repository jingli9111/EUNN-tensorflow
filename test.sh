echo ---------EURNN Ivan---------------
python2.7 copying_task.py EURNNIvan -T 500 -I $1 -F False -C True -B 128 -H 128 -L 4

echo ---------EURNN Orig---------------
python2.7 copying_task.py EURNN -T 500 -I $1 -F False -C True -B 128 -H 128 -L 4
