'''
tf.summary.scalar:
It's for writing the values of a scalar tensor that changes over time or iterations. 
In the case of neural networks it's usually used to monitor the changes of 
loss function or classification accuracy.
Let's have a look at another simple example to get the point.
This example generates 100 random values from a standard normal dist.'''
import tensorflow as tf
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell

# create the scalar variable
x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

# step 1:create the scalar summary
first_summary = tf.summary.scalar(name='My_first_scalar_summary', tensor=x_scalar)

init = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
    # step 2: creating the writer inside the session
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for step in range(100):
        # loop over several initializations of the variable
        sess.run(init)
        # step 3: evaluate the scalar summary
        summary = sess.run(first_summary)
        # step 4: add the summary to the writer (i.e. to the event file)
        writer.add_summary(summary, step)
    print('Done with writing the scalar summary')
# Now run the tensorboard 