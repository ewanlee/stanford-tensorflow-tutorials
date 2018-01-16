""" Starter code for logistic regression model to solve OCR task
with MNIST in TensorFlow
MNIST dataset: yann.lecun.com/exdb/mnist/
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES']='3'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 100
keep_prob = 0.6

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('./data/not_mnist', one_hot=True)

# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9.
# Features are of the type float, and labels are of the type int
X = tf.placeholder(tf.float32, shape=[batch_size, 784], name='image')
Y = tf.placeholder(tf.float32, shape=[batch_size, 10], name='label')
p = tf.placeholder(tf.float32, name='keep_prob')

# Step 3: create weights and bias
# weights and biases are initialized to 0
# shape of w depends on the dimension of X and Y so that Y = X * w + b
# shape of b depends on Y
w1 = tf.get_variable(shape=[784, 400], name='weights_1', dtype=tf.float32)
b1 = tf.Variable(tf.zeros([1, 400]), name='biases_1', dtype=tf.float32)
w2 = tf.get_variable(shape=[400, 400], name='weights_2', dtype=tf.float32)
b2 = tf.Variable(tf.zeros([1, 400]), name='biases_2', dtype=tf.float32)
w3 = tf.get_variable(shape=[400, 10], name='weights_3', dtype=tf.float32)
b3 = tf.Variable(tf.zeros([1, 10]), name='biases_3', dtype=tf.float32)

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE
def drop_connect(w, p=0.4):
    return tf.nn.dropout(w, keep_prob=p) * p

w1_dropped = drop_connect(w1, p)
out = tf.nn.relu(tf.add(tf.matmul(X, w1_dropped), b1))

w2_dropped = drop_connect(w2, p)
out = tf.nn.relu(tf.add(tf.matmul(out, w2_dropped), b2))

w3_dropped = drop_connect(w3, p)
logits = tf.add(tf.matmul(out, w3_dropped), b3, name='logits')

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')
tf.summary.scalar('mean_loss', loss)

# Step 6: define training op
# using gradient descent to minimize loss
optimize = tf.train.AdamOptimizer().minimize(loss)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('./graphs/e02/logistic_reg/train', sess.graph)
    test_writer = tf.summary.FileWriter('./graphs/e02/logistic_reg/test', sess.graph)

    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples/batch_size)
    train_steps = 0

    for i in range(n_epochs): # train the model n_epochs times
        total_loss = 0

        for _ in range(n_batches):
            train_steps += 1
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            # TO-DO: run optimizer + fetch loss_batch
            _, loss_batch, train_summary = sess.run([optimize, loss, merged], feed_dict={X: X_batch, Y: Y_batch, p: keep_prob})
            train_writer.add_summary(train_summary, train_steps)
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!') # should be around 0.35 after 25 epochs

    # test the model
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(

    n_batches = int(mnist.test.num_examples/batch_size)
    test_steps = 0
    total_correct_preds = 0

    for i in range(n_batches):
        test_steps += 1
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch, test_summary = sess.run([accuracy, merged], feed_dict={X: X_batch, Y:Y_batch, p: 1.0})
        test_writer.add_summary(test_summary, test_steps)
        total_correct_preds += accuracy_batch

    print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
