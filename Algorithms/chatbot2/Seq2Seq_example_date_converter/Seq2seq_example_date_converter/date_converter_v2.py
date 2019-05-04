import pandas as pd
import numpy as np
import tensorflow as tf
import time

data = pd.read_csv('date_data.csv')

# Inputs for the encoder (x) and decoder (y)
x = data['raw_dates'].values.tolist()
y = data['cleaned_dates'].values.tolist()

# Padding the inputs for encoder (x) with <PAD>
u_characters = set(' '.join(x))
char2numX = dict(zip(u_characters, range(len(u_characters))))

u_characters = set(' '.join(y))
char2numY = dict(zip(u_characters, range(len(u_characters)))) 

char2numX['<PAD>'] = len(char2numX)
num2charX = dict(zip(char2numX.values(), char2numX.keys()))
max_len = max([len(date) for date in x])

# Convert x to numbers and prepend with '<PAD>' 
x = [[char2numX['<PAD>']]*(max_len - len(date)) +[char2numX[x_] for x_ in date] for date in x]

# Lets check a sample padded date in x
print(''.join([num2charX[x_] for x_ in x[4]]))

# Convert x to numpy array
x = np.array(x)


# Adding '<GO>' to the input for the decoder
char2numY['<GO>'] = len(char2numY)
num2charY = dict(zip(char2numY.values(), char2numY.keys()))

# Convert y to numbers and prepend with '<GO>
y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y]

# Lets check a sample <GO> padded date in y
print(''.join([num2charY[y_] for y_ in y[4]]))

# Convert y to numpy array
y = np.array(y)

x_seq_length = len(x[0])
y_seq_length = len(y[0])- 1

# Define a function to generate random batches of x and y of the size batch_size
def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x)) # creates a list of random numbers in the range 0 to len(x)
    start = 0
    x = x[shuffle] # select a particular random x & y
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size]
        start += batch_size
        

# Creating the model structure
batch_size = 128
nodes = 32
embed_size = 10
bidirectional = True

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Placeholders for tensors which we will feed the data into graph
inputs = tf.placeholder(tf.int32, (None, x_seq_length), 'inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')

# Embedding layers
# Get an embed_size dense vector representation of each character in x & y
# during training time by uniformly & randomly initializing the embedding weighs 
# http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
input_embedding = tf.Variable(tf.random_uniform((len(char2numX), embed_size), -1.0, 1.0), name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
# Lookup for embedding representation of each input character in input_embedding
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs) # similarly for output

with tf.variable_scope("encoding") as encoding_scope: # Used for parameter sharing - 

    if not bidirectional:
        
        # Regular approach with LSTM units
        lstm_enc = tf.contrib.rnn.LSTMCell(nodes) # Creates a single LSTM green box with nodes number of neurons
        _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32) # http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
    else:
        
        # Using a bidirectional LSTM architecture instead
        # http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
        enc_fw_cell = tf.contrib.rnn.LSTMCell(nodes)
        enc_bw_cell = tf.contrib.rnn.LSTMCell(nodes)

        ((enc_fw_out, enc_bw_out) , (enc_fw_final, enc_bw_final)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_fw_cell,
                                                        cell_bw=enc_bw_cell, inputs=date_input_embed, dtype=tf.float32)
        enc_fin_c = tf.concat((enc_fw_final.c , enc_bw_final.c),1)
        enc_fin_h = tf.concat((enc_fw_final.h , enc_bw_final.h),1)
        last_state = tf.contrib.rnn.LSTMStateTuple(c=enc_fin_c , h=enc_fin_h)
    
    
with tf.variable_scope("decoding") as decoding_scope:
    
    if not bidirectional:      
        lstm_dec = tf.contrib.rnn.LSTMCell(nodes)    
    else:
        lstm_dec = tf.contrib.rnn.LSTMCell(2*nodes)
    
    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, initial_state=last_state)

        

logits = tf.layers.dense(dec_outputs, units=len(char2numY), use_bias=True) # Creates a len(char2numY) length output layer by implementing (wx+b) 
    
#connect outputs to 
with tf.name_scope("optimization"):
    # Loss function for the sequence of logits
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length])) # third argument is for weights
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)
    
# Training the seq2seq model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

sess.run(tf.global_variables_initializer())
epochs = 10
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
            feed_dict = {inputs: source_batch,
             outputs: target_batch[:, :-1],
             targets: target_batch[:, 1:]})
    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, 
                                                                      accuracy, time.time() - start_time))

# Testing on test set
source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))

dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
for i in range(y_seq_length):
    batch_logits = sess.run(logits,
                feed_dict = {inputs: source_batch,
                 outputs: dec_input})
    prediction = batch_logits[:,-1].argmax(axis=-1)
    dec_input = np.hstack([dec_input, prediction[:,None]])
    
print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))

# Sample predictions
num_preds = 2
source_chars = [[num2charX[l] for l in sent if num2charX[l]!="<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in dec_input[:num_preds, 1:]]

for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in)+' => '+''.join(date_out))
