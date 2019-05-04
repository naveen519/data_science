# Saving the variables
'''
Lets see how to save the parameters into the disk and restore the saved parameters from the disk. 
The savable/restorable paramters of the network are Variables (i.e. weights and biases)
'''
# Example 1 - Save variables
'''
We will start with saving and restoring two variables in TensorFlow. 
We will create a graph with two variables. Let's create two variables a = [3 3] and b = [5 5 5]
'''
import tensorflow as tf
tf.reset_default_graph()
# create variables a and b
a = tf.get_variable("A", initializer=tf.constant(3, shape=[2]))
b = tf.get_variable("B", initializer=tf.constant(5, shape=[3]))
# initialize all of the variables
init_op = tf.global_variables_initializer()
# run the session
with tf.Session() as sess:
    # initialize all of the variables in the session
    sess.run(init_op)
    # run the session to get the value of the variable
    a_out, b_out = sess.run([a, b])
    print('a = ', a_out)
    print('b = ', b_out)
'''
In order to save the variables, we use the saver function using tf.train.Saver() in the graph. 
This function will find all the variables in the graph. 
We can see the list of all variables in _var_list. 
Let's create a saver object and take a look at the _var_list in the object
'''
# create saver object
saver = tf.train.Saver()
for i, var in enumerate(saver._var_list):
    print('Var {}: {}'.format(i, var))   

'''
Now that the saver object is created in the graph, in the session, we can call 
the saver.save() function to save the variables in the disk. We have to pass the 
created session (sess) and the path to the file that we want to save the variables  
'''
# run the session
with tf.Session() as sess:
    # initialize all of the variables in the session
    sess.run(init_op)

    # save the variable in the disk
    saved_path = saver.save(sess, './saved_variable')
    print('model saved in {}'.format(saved_path))  
    
'''If you check your working directory, you will notice that 3 new files have been created 
with the name saved_variable in them. 
.data: Contains variable values

.meta: Contains graph structure

.index: Identifies checkpoints'''
import os
for file in os.listdir('.'):
    if 'saved_variable' in file:
        print(file)

###############################################################################
# Restoring variables
'''
Now that all the things that you need is saved in the disk, you can load your 
saved variables in the session using saver.restore()
'''
# run the session
with tf.Session() as sess:
    # restore the saved vairable
    saver.restore(sess, './saved_variable') # instead of initializing the variables in the session, we restore them from the disk
    # print the loaded variable
    a_out, b_out = sess.run([a, b])
    print('a = ', a_out)
    print('b = ', b_out)
    
    