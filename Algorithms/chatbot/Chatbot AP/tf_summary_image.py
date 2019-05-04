'''
tf.summary.image:
It's for for writing and visualizing tensors as images. In the case of neural networks, 
this is usually used for tracking the images that are either fed to the network (say in each batch) 
or the images generated in the output (such as the reconstructed images in an autoencoder; 
or the fake images made by the generator model of a Generative Adverserial Network). 
However, in general, this can be used for plotting any tensor. For example, you can 
visualize a weight matrix of size 30x40 as an image of 30x40 pixels.

An image summary can be created like:
tf.summary.image(name, tensor, max_outputs=3)
  
where name is the name for the generated node (i.e. operation), 
tensor is the desired tensor to be written as an image summary and max_outputs 
is the maximum number of elements from tensor to generate images for. 
The below example is used to plot images for two variables:
1. Of size 30x10 as 3 grayscale images of size 10x10
2. Of size 50x30 as 5 color images of size 10x10'''

import tensorflow as tf
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell

# create the variables
w_gs = tf.get_variable('W_Grayscale', shape=[30, 10], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
w_c = tf.get_variable('W_Color', shape=[50, 30], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

# step 0: reshape it to 4D-tensors
w_gs_reshaped = tf.reshape(w_gs, (3, 10, 10, 1)) # [batch_size, height, width, channels]
w_c_reshaped = tf.reshape(w_c, (5, 10, 10, 3)) # [batch_size, height, width, channels]

# step 1: create the summaries
gs_summary = tf.summary.image('Grayscale', w_gs_reshaped)
c_summary = tf.summary.image('Color', w_c_reshaped, max_outputs=5)

# step 2: merge all summaries
merged = tf.summary.merge_all()

# create the op for initializing all variables
init = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
    # step 3: creating the writer inside the session
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # initialize all variables
    sess.run(init)
    # step 4: evaluate the merged op to get the summaries
    summary = sess.run(merged)
    # step 5: add summary to the writer (i.e. to the event file) to write on the disc
    writer.add_summary(summary)
    print('Done writing the summaries')
 