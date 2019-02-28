import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.slim as slim
from lib import *

BATCH_SIZE = 8
BETA = 0.01

class Model:
	
	def __init__(self, trainData):
		self.trainData = trainData[BATCH_SIZE:]
		self.testData = trainData[:BATCH_SIZE]
		
		# Define networks
		self.var_com = make_dict(5, 32, 'c', 1, 1)
		self.var_gen = make_dict(10, 32, 'g', 1, 1)
		
		# Define target image
		self.img_ans = tf.placeholder(tf.float32, [None, 513, 513])
		
		# Define graph for C(C-G graph)
		self.img_C_input = tf.placeholder(tf.float32, [None, 513, 513])
		self.img_C_com = self.com_net(self.img_C_input, self.var_com['weights'], self.var_com['biases'])
		self.img_C_gen = self.gen_net(self.img_C_com, self.var_gen['weights'], self.var_gen['biases'])
		self.img_C_final = self.img_C_com + self.img_C_gen
		self.C_loss = tf.reduce_mean(tf.square(self.img_ans - self.img_C_final))
		
		# Define graph for G(G graph)
		self.img_G_input = tf.placeholder(tf.float32, [None, 513, 513])
		self.img_G_gen = self.gen_net(self.img_G_input, self.var_gen['weights'], self.var_gen['biases'])
		self.img_G_final = self.img_G_input + self.img_G_gen
		self.G_loss = tf.reduce_mean(tf.square(self.img_ans - self.img_G_final))
	
		# Define optimizers
		self.C_lr, self.G_lr = 0.001, 0.005
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.C_opt = tf.train.AdamOptimizer(learning_rate = self.C_lr).minimize(self.C_loss)
			self.G_opt = tf.train.AdamOptimizer(learning_rate = self.G_lr).minimize(self.G_loss)
		
		config = tf.ConfigProto()
		self.sess = tf.InteractiveSession(config = config)
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(tf.global_variables())
				
	def com_net(self, x, weights, biases):
		x = tf.reshape(x, [-1, 513, 513, 1])
		x = conv2d(x, weights['wc1'], biases['bc1'], act_func = 'LReLU', use_bn = False)
		for i in range(2, 2):
			name_w = 'wc'+str(i)
			name_b = 'bc'+str(i)
			x = conv2d(x, weights[name_w], biases[name_b], act_func = 'LReLU')   
		res = conv2d(x, weights['wcx'], biases['bcx'], act_func='Sigmoid', use_bn = False)
		res = tf.reshape(res, [-1, 513, 513])
		return res

	def gen_net(self, x, weights, biases):
		x = tf.reshape(x, [-1, 513, 513, 1])
		x = conv2d(x, weights['wc1'], biases['bc1'], act_func = 'LReLU', use_bn = False) 
		for i in range(1, 6//2):
			x_input = x
			name_w = 'wc'+str(2*i)
			name_b = 'bc'+str(2*i)
			x_input = conv2d(x_input, weights[name_w], biases[name_b], act_func='LReLU')
			name_w = 'wc'+str(2*i+1)
			name_b = 'bc'+str(2*i+1)
			x_input = conv2d(x_input, weights[name_w], biases[name_b], act_func='None')
			x = leaky_relu(x_input + x)
		res = conv2d(x, weights['wcx'], biases['bcx'], act_func='TanH', use_bn = False)
		res = tf.reshape(res, [-1, 513, 513])
		return res
	
	def train(self):
		# Create batch
		data = []
		for _ in range(BATCH_SIZE):
			pos = random.randrange(0, len(self.trainData))
			data.append(self.trainData[pos])

		# Train C
		_, C_loss = self.sess.run([self.C_opt, self.C_loss], feed_dict = {self.img_C_input: data, self.img_ans: data})
		
		# Train G
		img_G_input = self.sess.run(self.img_C_com, feed_dict = {self.img_C_input: data})
		_, G_loss = self.sess.run([self.G_opt, self.G_loss], feed_dict = {self.img_G_input: img_G_input, self.img_ans: data})
		return C_loss, G_loss

	def test(self):
		data = self.testData
		
		# Test C
		_, C_loss = self.sess.run([self.C_opt, self.C_loss], feed_dict = {self.img_C_input: data, self.img_ans: data})
		
		# Test G
		img_G_input = self.sess.run(self.img_C_com, feed_dict = {self.img_C_input: data})
		_, G_loss = self.sess.run([self.G_opt, self.G_loss], feed_dict = {self.img_G_input: img_G_input, self.img_ans: data})
		return C_loss, G_loss


	def save(self, name):
		self.saver.save(self.sess, name)

	def load(self, name):
		self.saver.restore(self.sess, name)
	
	
