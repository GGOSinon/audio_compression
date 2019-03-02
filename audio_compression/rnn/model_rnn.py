import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.slim as slim
from lib import *
from PIL import Image

BATCH_SIZE = 8
BETA = 0.01
T, F = 256, 256
F_encode = 127
sz_encode = 2048

xaiver = tf.contrib.layers.xavier_initializer()
class Model:
	
	def __init__(self, trainData):
		self.trainData = trainData[BATCH_SIZE:]
		self.testData = trainData[:BATCH_SIZE]
		
		# Define networks
		self.var_com = {
			'rnn': 0,
			'weights': {
				'wd1': tf.get_variable('wd1c', [F_encode * T, sz_encode]),
				'wd2': tf.get_variable('wd2c', [sz_encode, sz_encode])
			},
			'biases':{
				'bd1': tf.get_variable('bd1c', [sz_encode]),
				'bd2': tf.get_variable('bd2c', [sz_encode])
			}
		}
		with tf.variable_scope('lstm1'):
			self.var_com['rnn'] = tf.nn.rnn_cell.BasicLSTMCell(F_encode)
		self.var_gen = {
			'rnn': 0,
			'weights': {
				'wd1': tf.get_variable('wd1g', [sz_encode, sz_encode]),
				'wd2': tf.get_variable('wd2g', [sz_encode, F_encode * T])
			},
			'biases':{
				'bd1': tf.get_variable('bd1g', [sz_encode]),
				'bd2': tf.get_variable('bd2g', [F_encode * T])
			}
		}
		with tf.variable_scope('lstm2'):
			self.var_gen['rnn'] = tf.nn.rnn_cell.BasicLSTMCell(F)

		# Define target image
		self.img_input = tf.placeholder(tf.float32, [None, T, F])
		self.img_ans = tf.placeholder(tf.float32, [None, T, F])
		
		# Define encoder network
		self.img_encode = self.com_net(self.img_input, self.var_com)
		
		# Define decoder network
		self.img_decode = self.gen_net(self.img_encode, self.var_gen)
		
		# Define loss and optimizers
		self.loss = tf.reduce_mean(tf.square(self.img_ans - self.img_decode))
		self.lr = tf.Variable(0.001)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.opt = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)
		
		# Define session
		config = tf.ConfigProto()
		self.sess = tf.InteractiveSession(config = config)
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(tf.global_variables())
			
	def com_net(self, x, network):
		rnn, weights, biases = network['rnn'], network['weights'], network['biases']
		x = tf.reshape(x, [-1, T, F])
		with tf.variable_scope('lstm1'):
			x, _ = tf.nn.dynamic_rnn(rnn, x, dtype = tf.float32)
		x = tf.reshape(x, [-1, F_encode * T])
		x = dense(x, weights['wd1'], biases['bd1'])
		x = dense(x, weights['wd2'], biases['bd2'], act_func = 'sigmoid', use_bn = False)
		x = tf.reshape(x, [-1, sz_encode])
		return x

	def gen_net(self, x, network):
		rnn, weights, biases = network['rnn'], network['weights'], network['biases']
		x = tf.reshape(x, [-1, sz_encode])
		x = dense(x, weights['wd1'], biases['bd1'])
		x = dense(x, weights['wd2'], biases['bd2'])
		x = tf.reshape(x, [-1, T, F_encode])
		#print(x.shape)
		with tf.variable_scope('lstm2'):
			x, _ = tf.nn.dynamic_rnn(rnn, x, dtype = tf.float32)
		x = tf.nn.tanh(x)
		x = tf.reshape(x, [-1, T, F])
		return x
	
	def train(self):
		# Create batch
		data = []
		for _ in range(BATCH_SIZE):
			pos = random.randrange(0, len(self.trainData))
			data.append(self.trainData[pos])
		
		_, loss = self.sess.run([self.opt, self.loss], feed_dict = {self.img_input: data, self.img_ans: data})
		return loss

	def test(self):
		data = self.testData
		_, loss = self.sess.run([self.opt, self.loss], feed_dict = {self.img_input: data, self.img_ans: data})
		return loss

	def save(self, name):
		self.saver.save(self.sess, name)

	def load(self, name):
		self.saver.restore(self.sess, name)
	
	
