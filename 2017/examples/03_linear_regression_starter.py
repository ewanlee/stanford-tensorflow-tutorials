""" Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from
the number of fire in the city of Chicago
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

import utils

DATA_FILE = 'data/fire_theft.xls'

# Phase 1: Assemble the graph
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
# Both have the type float32
X = tf.placeholder(dtype=tf.float32, shape=[None], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None], name='Y')

# Step 3: create weight and bias, initialized to 0
# name your variables w and b
weight = tf.Variable(initial_value=0.0, name='weight', dtype=tf.float32)
bias = tf.Variable(initial_value=0.0, name='bias', dtype=tf.float32)

# Step 4: predict Y (number of theft) from the number of fire
# name your variable Y_predicted
Y_predicted = tf.add(tf.multiply(X, weight), bias, name='Y_predicted')

# Step 5: use the square error as the loss function
# name your variable loss
loss = tf.reduce_mean(tf.losses.huber_loss(Y, Y_predicted), name='mean_huber_loss')
tf.summary.scalar('summary_loss', loss)

# Step 6: using gradient descent with learning rate of 0.001 to minimize loss
optimize = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

# Phase 2: Train our model
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs/03/linear_reg', sess.graph)
    # Step 7: initialize the necessary variables, in this case, w and b
    # TO - DO
    sess.run(init)

    # Step 8: train the model
    for i in range(100000): # run 100 epochs
        x = data[:, 0]
        y = data[:, 1]
        # Session runs optimizer to minimize loss and fetch the value of loss. Name the received value as l
        # TO DO: write sess.run()
        _, l, summary = sess.run([optimize, loss, merged], feed_dict={X: x, Y: y})
        writer.add_summary(summary, i)
        print("Epoch {0}: {1}".format(i, l))

    w, b = sess.run([weight, bias])

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()
