# Tensorboard basics
'''Tensorboard is a visualization tool that comes with Tensorflow. 
The computations you'll use TensorFlow for (like training a massive deep neural network) 
can be complex and confusing. Tensorboard is used to make it easier to 
understand, debug, and optimize TensorFlow programs. There are two main uses of Tensorboard:
1. Visualizing the Graph
2. Writing Summaries to Visualize Learning'''
# Example 1 - Tensorboard
import tensorflow as tf

# create graph
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
# launch the graph in a session
with tf.Session() as sess:
    print(sess.run(c))
    
'''
To visualize the program with TensorBoard, we need to write log files of the program. 
To write event files, we first need to create a writer for those logs, using this code:

writer = tf.summary.FileWriter([logdir], [graph]) 
 
where [logdir] is the folder where you want to store those log files. 
You can choose [logdir] to be something meaningful such as './graphs'. 
The second argument [graph] is the graph of the program we're working on. 
There are two ways to get the graph:

1. Call the graph using tf.get_default_graph(), which returns the default graph of the program
2. Set it as sess.graph which returns the session's graph (this requires us to already have created a session)
The second way is more common. Either way, make sure to create a writer only after you’ve defined your graph. 
Otherwise, the graph visualized on TensorBoard would be incomplete.

Let's add the writer to the first example and visualize the graph.
'''

tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell

# create graph
a = tf.constant(2, name = 'a')
b = tf.constant(3, name = 'b')
c = tf.add(a, b, name = 'addition')

# creating the writer out of the session
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

# launch the graph in a session
with tf.Session() as sess:
    # or creating the writer inside the session
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(c))

'''
Next, go to Terminal and make sure that the present working directory is the same
as where you ran your Python code. For example, here we can switch to the directory using

$ cd ~/Desktop/Chatbot AP

Then run:

$ tensorboard --logdir="./graphs" --port 6006

This will generate a link for you. ctrl+left click on that link 
(or simply copy it into your browser or just open your browser and go to 
http://localhost:6006/. This will show the TensorBoard page
'''

'''
Tensorboard is also used to visualize the model parameters (like weights and biases of a neural network),
metrics (like loss or accuracy value), and images (like input images to a network) 
using the Summary functionality.
Summary is a special TensorBoard operation that takes in a regular tensor 
and outputs the summarized data to your disk (i.e. in the event file). 
Basically, there are three main types of summaries:
1. tf.summary.scalar: used to write a single scalar-valued tensor (like classificaion loss or accuracy value)
2. tf.summary.histogram: used to plot histogram of all the values of a non-scalar tensor (like weight or bias matrices of a neural network)
3. tf.summary.image: used to plot images (like input images of a network, or generated output images of an autoencoder or a GAN)

Lets consider separate examples to understand these in details'''
 
