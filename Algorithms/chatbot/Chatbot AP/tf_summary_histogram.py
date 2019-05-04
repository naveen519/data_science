'''
tf.summary.histogram:
It's for plotting the histogram of the values of a non-scalar tensor. 
This gives us a view of how does the histogram (and the distribution) of the tensor values 
change over time or iterations. In the case of neural networks, 
it's commonly used to monitor the changes of weights and biases distributions. 
It's very useful in detecting irregular behavior of the network parameters 
(like when many of the weights shrink to almost zero or grow largely).
Let's have a look at another simple example to get the point.
This example generates 100 random values of a 30x40 matrix from a standard normal dist.'''
import tensorflow as tf
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell

# create the variables
x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
x_matrix = tf.get_variable('x_matrix', shape=[30, 40], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

# step 1: create the summaries
# A scalar summary for the scalar tensor
scalar_summary = tf.summary.scalar('My_scalar_summary', x_scalar)
# A histogram summary for the non-scalar (i.e. 2D or matrix) tensor
histogram_summary = tf.summary.histogram('My_histogram_summary', x_matrix)

init = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
    # step 2: creating the writer inside the session
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for step in range(100):
        # loop over several initializations of the variable
        sess.run(init)
        # step 3: evaluate the merged summaries
        summary1, summary2 = sess.run([scalar_summary, histogram_summary])
        # step 4: add the summary to the writer (i.e. to the event file) to write on the disc
        writer.add_summary(summary1, step)
        # repeat steps 4 for the histogram summary
        writer.add_summary(summary2, step)
    print('Done writing the summaries')
# Now run the tensorboard 