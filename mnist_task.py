from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

import numpy as np
import argparse
import tensorflow as tf

from EURNN import *

import random

from random import shuffle
from tensorflow.examples.tutorials.mnist import input_data

random.seed(2017)



def mnist_data(object, n_batch, ind, dataset):

	mnist = object
	if dataset == "train":
		xx, yy = mnist.train.next_batch(n_batch)
	elif dataset == "validation": 
		xx, yy = mnist.validation.next_batch(n_batch)
	elif dataset == "test": 
		xx, yy = mnist.test.next_batch(n_batch)

	step1 = np.array(xx)
	step2 = np.transpose(step1)
	step3 = [step2[i] for i in ind]
	xx = np.transpose(step3)

	x = []
	y = []
	for i in range(n_batch):
		x.append(xx[i].reshape((28*28, 1)))
		y.append(yy[i])
	
	shuffle_list = list(range(n_batch))
	shuffle(shuffle_list)
	
	x = np.array([x[i] for i in shuffle_list])
	y = np.array([y[i] for i in shuffle_list]).astype(np.int64)

	return x, y

def main(model, n_iter, n_batch, n_hidden, capacity, comp, FFT):

	# --- Set data params ----------------
	n_input = 1
	n_output = 10
	n_train = n_iter * n_batch
	n_val = 5000
	n_test = 10000

	n_steps = 28 * 28
	n_classes = 10


	# --- Create data --------------------

	ind = list(range(784))
	shuffle(ind)



	mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)



	# --- Create graph and compute gradients ----------------------
	x = tf.placeholder("float", [None, n_steps, n_input])
	y = tf.placeholder("int64", [None])
	

	# --- Input to hidden layer ----------------------
	if model == "LSTM":
		cell = core_rnn_cell_impl.BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
	elif model == "EURNN":
		cell = EURNNCell(n_hidden, capacity, FFT, comp)
		if comp:
			hidden_out_comp, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.complex64)
			hidden_out = tf.real(hidden_out_comp)
		else:
			hidden_out, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

	# --- Hidden Layer to Output ----------------------
	V_init_val = np.sqrt(6.)/np.sqrt(n_output + n_input)

	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_classes], \
			dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
	V_bias = tf.get_variable("V_bias", shape=[n_classes], \
			dtype=tf.float32, initializer=tf.constant_initializer(0.01))

	hidden_out_list = tf.unstack(hidden_out, axis=1)
	temp_out = tf.matmul(hidden_out_list[-1], V_weights)
	output_data = tf.nn.bias_add(temp_out, V_bias) 
	
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_data, labels=y))
	correct_pred = tf.equal(tf.argmax(output_data, 1), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



	# --- Initialization --------------------------------------------------
	optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9).minimize(cost)
	init = tf.global_variables_initializer()

	# --- Training Loop ---------------------------------------------------------------


	with tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=False)) as sess:

		sess.run(init)

		step = 0
		steps = []
		losses = []
		accs = []

		while step < n_iter:

			batch_x, batch_y = mnist_data(mnist, n_batch, ind, "train")

			loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
			acc = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})

			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
			print(" Iter: " + str(step) + ", Minibatch Loss= " + \
			  "{:.6f}".format(loss) + ", Training Accuracy= " + \
			  "{:.5f}".format(acc))


			if step % 500 == 499:
				val_x, val_y = mnist_data(mnist, n_val, ind, "validation")
				val_index = 0
				val_acc_list = []
				val_loss_list = []
				for i in range(50):
					batch_x = val_x[val_index: val_index + 100]
					batch_y = val_y[val_index: val_index + 100]
					val_index += 100
					val_acc_list.append(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
					val_loss_list.append(sess.run(cost, feed_dict={x: batch_x, y: batch_y}))
				val_acc = np.mean(val_acc_list)
				val_loss = np.mean(val_loss_list)
				print("Iter " + str(step) + ", Validation Loss= " + \
				  "{:.6f}".format(val_loss) + ", Validation Accuracy= " + \
				  "{:.5f}".format(val_acc))

				steps.append(step)
				losses.append(val_loss)
				accs.append(val_acc)



			step += 1

			
				

		print("Optimization Finished!")

		
		# --- test ----------------------

		test_x, test_y = mnist_data(mnist, n_test, ind, "test")
		test_index = 0
		test_acc_list = []
		test_loss_list = []

		for i in range(100):
			batch_x = test_x[test_index: test_index + 100]
			batch_y = test_y[test_index: test_index + 100]

			test_index += 100
			test_acc_list.append(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
			test_loss_list.append(sess.run(cost, feed_dict={x: batch_x, y: batch_y}))

		test_acc = np.mean(test_acc_list)
		test_loss = np.mean(test_loss_list)

		print("Test result: Loss= " + "{:.6f}".format(test_loss) + \
					", Accuracy= " + "{:.5f}".format(test_acc))


				



if __name__=="__main__":
	parser = argparse.ArgumentParser(
		description="Pixel-Permuted MNIST Task")
	parser.add_argument("model", default='LSTM', help='Model name: LSTM, EURNN')
	parser.add_argument('--n_iter', '-I', type=int, default=50000, help='training iteration number')
	parser.add_argument('--n_batch', '-B', type=int, default=128, help='batch size')
	parser.add_argument('--n_hidden', '-H', type=int, default=128, help='hidden layer size')
	parser.add_argument('--capacity', '-L', type=int, default=2, help='Tunable style capacity, only for EURNN, default value is 2')
	parser.add_argument('--comp', '-C', type=str, default="False", help='Complex domain or Real domain. Default is False: real domain')
	parser.add_argument('--FFT', '-F', type=str, default="False", help='FFT style, only for EURNN, default is False')

	args = parser.parse_args()
	dict = vars(args)

	for i in dict:
		if (dict[i]=="False"):
			dict[i] = False
		elif dict[i]=="True":
			dict[i] = True
		
	kwargs = {	
				'model': dict['model'],
				'n_iter': dict['n_iter'],
				'n_batch': dict['n_batch'],
				'n_hidden': dict['n_hidden'],
				'capacity': dict['capacity'],
				'comp': dict['comp'],
				'FFT': dict['FFT'],
			}

	main(**kwargs)
