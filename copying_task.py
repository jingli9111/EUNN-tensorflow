from __future__ import absolute_import
from __future__ import division
from __future__	import print_function

import numpy as np
import argparse
import tensorflow as tf

from EURNN import *


def copying_data(T, n_data, n_sequence):
	seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
	zeros1 = np.zeros((n_data, T-1))
	zeros2 = np.zeros((n_data, T))
	marker = 9 * np.ones((n_data, 1))
	zeros3 = np.zeros((n_data, n_sequence))

	x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
	y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int64')
	
	return x, y

def main(model, T, n_iter, n_batch, n_hidden, capacity, comp, FFT):

	# --- Set data params ----------------
	n_input = 10
	n_output = 9
	n_sequence = 10
	n_train = n_iter * n_batch
	n_test = n_batch

	n_input = 10
	n_steps = T+20
	n_classes = 9


  	# --- Create data --------------------

	train_x, train_y = copying_data(T, n_train, n_sequence)
	test_x, test_y = copying_data(T, n_test, n_sequence)


	# --- Create graph and compute gradients ----------------------
	x = tf.placeholder("int32", [None, n_steps])
	y = tf.placeholder("int64", [None, n_steps])
	
	input_data = tf.one_hot(x, n_input, dtype=tf.float32)



	# --- Input to hidden layer ----------------------
	if model == "LSTM":
		cell = core_rnn_cell_impl.BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
		hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
	elif model == "EURNN":
		cell = EURNNCell(n_hidden, capacity, FFT, comp)
		if comp:
			hidden_out_comp, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.complex64)
			hidden_out = tf.real(hidden_out_comp)
		else:
			hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)

	# --- Hidden Layer to Output ----------------------
	V_init_val = np.sqrt(6.)/np.sqrt(n_output + n_input)

	V_weights = tf.get_variable("V_weights", shape = [n_hidden, n_classes], \
			dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
	V_bias = tf.get_variable("V_bias", shape=[n_classes], \
			dtype=tf.float32, initializer=tf.constant_initializer(0.01))

	hidden_out_list = tf.unstack(hidden_out, axis=1)
	temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
	output_data = tf.nn.bias_add(tf.transpose(temp_out, [1,0,2]), V_bias) 

	# --- evaluate process ----------------------
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_data, labels=y))
	correct_pred = tf.equal(tf.argmax(output_data, 2), y)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	# --- Initialization ----------------------
	optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9).minimize(cost)
	init = tf.global_variables_initializer()



	# --- baseline ----------------------
	baseline = np.log(8) * 10/(T+20)
	print("Baseline is " + str(baseline))


	# --- Training Loop ----------------------


	with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:

		sess.run(init)

		step = 0
		steps = []
		losses = []
		accs = []

		while step < n_iter:
			batch_x = train_x[step * n_batch : (step+1) * n_batch]
			batch_y = train_y[step * n_batch : (step+1) * n_batch]

			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

			acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

			print("Iter " + str(step) + ", Minibatch Loss= " + \
				  "{:.6f}".format(loss) + ", Training Accuracy= " + \
				  "{:.5f}".format(acc))

			steps.append(step)
			losses.append(loss)
			accs.append(acc)
			step += 1


		print("Optimization Finished!")


		
		# --- test ----------------------

		sess.run(optimizer, feed_dict={x: test_x, y: test_y})
		test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
		test_loss = sess.run(cost, feed_dict={x: test_x, y: test_y})
		print("Test result: Loss= " + "{:.6f}".format(test_loss) + \
					", Accuracy= " + "{:.5f}".format(test_acc))


if __name__=="__main__":
	parser = argparse.ArgumentParser(
		description="Copying Memory Task")
	parser.add_argument("model", default='LSTM', help='Model name: LSTM, EURNN')
	parser.add_argument('-T', type=int, default=1000, help='Copying Problem delay')
	parser.add_argument('--n_iter', '-I', type=int, default=5000, help='training iteration number')
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
				'T': dict['T'],
				'n_iter': dict['n_iter'],
			  	'n_batch': dict['n_batch'],
			  	'n_hidden': dict['n_hidden'],
			  	'capacity': dict['capacity'],
			  	'comp': dict['comp'],
			  	'FFT': dict['FFT'],
			}

	main(**kwargs)
