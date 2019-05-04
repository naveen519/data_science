# Tensorflow basics
import tensorflow as tf
a = 2
b = 3
c = tf.add(a, b, name = 'Add')
print(c) # This only creates the graph, but doesnot print 5 due to lazy execution

# Create a session to execute the graph
sess = tf.Session() # create a session object
print(sess.run(c)) # this executes the graph
sess.close()

# This can also be written as follows (without explicitly closing the session)
with tf.Session() as sess:
    print(sess.run(c))
    
# Example 2
import tensorflow as tf
x = 2
y = 3
add_op = tf.add(x, y, name='Add')
mul_op = tf.multiply(x, y, name='Multiply')
pow_op = tf.pow(add_op, mul_op, name='Power')
useless_op = tf.multiply(x, add_op, name='Useless')

with tf.Session() as sess:
    pow_out, useless_out = sess.run([pow_op, useless_op])
    
'''if we fetch the pow_op operation, it will first run the add_op and mul_op to get their output tensor
and then run pow_op on them to compute the required output value. In other words useless_op 
will not be executed as it's output tensor is not used in executing the pow_op operation.'''

'''This is one of the advantages of defining a graph and running a session on it! 
It helps running only the required operations of the graph and skip the rest. 
This specially saves a significant amount of time for us when dealing with huge networks 
with hundreds and thousands of operations.'''

###############################################################################
# Tensorflow data types - Constants, Variables and Placeholders
# Example 1 - Constants
# create graph
a = tf.constant(2, name = 'A')
b = tf.constant(3, name = 'B')
c = tf.add(a, b, name = 'Sum')
# launch the graph in a session
with tf.Session() as sess:
    print(sess.run(c))
    
'''Constants can also be defined with different types (integer, float, etc.) 
and shapes (vectors, matrices, etc.). The next example has one constant with type 32bit float 
and another constant with shape 2X2'''
# Example 2 - Constants
s = tf.constant(2.3, name='scalar', dtype=tf.float32)
m = tf.constant([[1, 2], [3, 4]], name='matrix')
# launch the graph in a session
with tf.Session() as sess:
    print(sess.run(s))
    print(sess.run(m))

# Variables
'''Variables are data types whose values updates during execution. This makes them ideal candidates
for network parameters such as weights and biases. Variables can be initialized as tf.Variable()
or tf.get_variable().'''
# Example 1 - Variable
a = tf.get_variable(name="var_1", initializer=tf.constant(2))
b = tf.get_variable(name="var_2", initializer=tf.constant(3))
c = tf.add(a, b, name="Add1")

# launch the graph in a session
with tf.Session() as sess:
    # now let's evaluate their value
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
# This will give an error - because we didn't initialize the variables. Lets correct this as below:

# create graph
a = tf.get_variable(name="A", initializer=tf.constant(2))
b = tf.get_variable(name="B", initializer=tf.constant(3))
c = tf.add(a, b, name="Add")
# add an Op to initialize global variables
init_op = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
    # run the variable initializer operation
    sess.run(init_op)
    # now let's evaluate their value
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))

'''Variables are usually used for weights and biases in neural networks.
Weights are usually initialized from a normal distribution using tf.truncated_normal_initializer().
Biases are usually initialized from zeros using tf.zeros_initializer().

Let's look at another example of creating weight and bias variables with proper initialization.
Here we create the weight and bias matrices for a fully-connected layer with 2 neuron 
to another layer with 3 neuron. In this scenario, the weight and bias variables
must be of size [2, 3] and 3 respectively.'''
# Example 2 - Variable
# create graph
weights = tf.get_variable(name="W", shape=[2,3], initializer=tf.truncated_normal_initializer(stddev=0.01))
biases = tf.get_variable(name="b", shape=[3], initializer=tf.zeros_initializer())

# add an Op to initialize global variables
init_op = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
    # run the variable initializer
    sess.run(init_op)
    # now we can run our operations
    W, b = sess.run([weights, biases])
    print('weights = {}'.format(W))
    print('biases = {}'.format(b))
    
# Placeholders
'''Placeholders are like variables, with the only difference 
that we asign data to them in a future / execution time. If we have inputs to our network 
that depend on some external data and we don't want our graph to depend on any real value 
while developing the graph, placeholders are the datatype we need. 
In fact, we can build the graph without any data. 
Therefore, placeholders don't need any initial value; only a datatype (such as float32)
and a tensor shape so the graph still knows what to compute 
with even though it doesn't have any stored values yet'''
# Example 1 - Placeholder
a = tf.constant([5, 5, 5], tf.float32, name='A')
b = tf.placeholder(tf.float32, shape=[3], name='B')
c = tf.add(a, b, name="Add")

with tf.Session() as sess:
      print(sess.run(c)) 

'''This will give error because the placeholder is empty and there is no way 
to add an empty tensor to a constant tensor in the add operation. 
To solve this, we need to feed an input value to the tensor "b". 
It can be done by creating a dictionary ("d" in the following code) 
whose key(s) are the placeholders and their values are the desired value 
to be passed to the placeholder(s), and feeding it to an argument called "feed_dict"'''
a = tf.constant([5, 5, 5], tf.float32, name='A')
b = tf.placeholder(tf.float32, shape=[3], name='B')
c = tf.add(a, b, name="Add")

with tf.Session() as sess:
    # create a dictionary:
    d = {b: [1, 2, 3]}
    # feed it to the placeholder
    print(sess.run(c, feed_dict=d))      

###############################################################################
# Lets now try to implement the graph for one layer network with one hidden layer and 200 hidden units (neurons)
# for the MNIST dataset that contains 784 neurons in the input layer
# import the tensorflow library
import numpy as np

# create the input placeholder
X = tf.placeholder(tf.float32, shape=[None, 784], name="X") # None to feed variable batch-length inputs during test, 784 is the input size (28 pixel x 28 pixel)

# create network parameters
weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
W = tf.get_variable(name="Weight", dtype=tf.float32, shape=[784, 200], initializer=weight_initer)
bias_initer = tf.constant(0., shape=[200], dtype=tf.float32)
b = tf.get_variable(name="Bias", dtype=tf.float32, initializer=bias_initer)

# create MatMul node
x_w = tf.matmul(X, W, name="MatMul")
# create Add node
x_w_b = tf.add(x_w, b, name="Add")
# create ReLU node
h = tf.nn.relu(x_w_b, name="ReLU") 

# Add an Op to initialize variables
init_op = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
    # initialize variables
    sess.run(init_op)
    # create the dictionary:
    d = {X: np.random.rand(100, 784)}
    # feed it to placeholder a via the dict 
    print(sess.run(h, feed_dict=d))    


 
 