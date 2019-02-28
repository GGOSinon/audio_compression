import tensorflow as tf

def leaky_relu(x, alpha=0.2):
	return tf.maximum(x, alpha * x)
			 
# Conv2D wrapper, with bias and various activations
def conv2d(x, W, b, stride = 1, act_func = 'LReLU', use_bn = True, padding = 'SAME'):
	strides = [1, stride, stride, 1]
	sz = [[0,0],[1,1],[1,1],[0,0]]
	if padding == 'SAME':
		x = tf.nn.conv2d(x, W, strides, padding='SAME')
	if padding == 'VALID':
		x = tf.pad(x, sz, mode='SYMMETRIC')
		x = tf.nn.conv2d(x, W, strides, padding='VALID')
	x = tf.nn.bias_add(x, b)
	if act_func == 'LReLU': x = leaky_relu(x)
	if act_func == 'ReLU': x = tf.nn.relu(x)
	if act_func == 'TanH': x = tf.nn.tanh(x)
	if act_func == 'PReLU': x = leaky_relu(x)#tf.keras.layers.PReLU(x)
	if act_func == 'Sigmoid': x = tf.nn.sigmoid(x)
	if act_func == 'Softmax': x = tf.nn.softmax(x)
	if act_func == 'None': pass
	if use_bn: return tf.layers.batch_normalization(x)
	else: return x
	 
def make_grad(s, e, name):
	var = tf.get_variable(name, [3, 3, s, e], initializer=tf.contrib.layers.xavier_initializer())
	#var_list.append(var)
	return var

def make_bias(x, name):
	var = tf.get_variable(name, [x], initializer=tf.contrib.layers.xavier_initializer())
	#var_list.append(var)
	return var

def make_dict(num_layer, num_filter, end_str, s_filter = 1, e_filter = 1):							     
	result = {}
	weights = {}
	biases = {}

	weights['wc1'] = make_grad(s_filter,num_filter,"w1"+end_str)
	for i in range(2, num_layer):
		index = 'wc' + str(i)
		name = 'w' + str(i) + end_str
		weights[index] = make_grad(num_filter, num_filter, name)
	weights['wcx'] = make_grad(num_filter,e_filter,"wx"+end_str)
	
	biases['bc1'] = make_bias(num_filter,"b1"+end_str)
	for i in range(2, num_layer):
		index = 'bc' + str(i)
		#print(index)
		name = 'b' + str(i) + end_str
		biases[index] = make_bias(num_filter, name)
	biases['bcx'] = make_bias(e_filter,"bx"+end_str)

	result['weights'] = weights
	result['biases'] = biases
	return result
