import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import tensorflow as tf
import time
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import shutil

tf.set_random_seed(42)
np.random.seed(42)

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 500

# Step 1: Read in data
data = pd.read_csv('./data/heart/heart.csv').as_matrix()
famhist = data[:, 4]
data_without_famhist = np.delete(data, 4, 1)

lb = preprocessing.LabelBinarizer()
famhist_bin = lb.fit_transform(famhist)
ohenc = preprocessing.OneHotEncoder()
famhist_onehot = ohenc.fit_transform(famhist_bin).toarray()

data = np.c_[data_without_famhist, famhist_onehot]
features = data[:, :-1]
labels = ohenc.fit_transform(data[:, -1].reshape(-1, 1)).toarray()
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2)

# Step 2: create placeholders for features and labels
X = tf.placeholder(tf.float32, shape=[None, 10], name='feature')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='label')

# Step 3: create weights and bias
# weights and biases are initialized to 0
# shape of w depends on the dimension of X and Y so that Y = X * w + b
# shape of b depends on Y
w = tf.Variable(tf.random_normal(shape=[10, 2], stddev=0.01), name='weights', dtype=tf.float32)
b = tf.Variable(tf.zeros([1, 2]), name='biases', dtype=tf.float32)

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE
logits = tf.add(tf.matmul(X, w), b, name='logits')

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')
tf.summary.scalar('mean_loss', loss)

preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))# need numpy.count_nonzero(boolarr) :(

# Step 6: define training op
# using gradient descent to minimize loss
optimize = tf.train.AdamOptimizer().minimize(loss)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    shutil.rmtree('./graphs/e03/heart/')
    train_writer = tf.summary.FileWriter('./graphs/e03/heart/train', sess.graph)
    test_writer = tf.summary.FileWriter('./graphs/e03/heart/test', sess.graph)

    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    train_steps = 0
    for i in range(n_epochs): # train the model n_epochs times
        for j in range(0, len(X_train), batch_size):
            train_steps += 1
            X_batch, Y_batch = X_train[j : j + batch_size], Y_train[j : j + batch_size]
            # TO-DO: run optimizer + fetch loss_batch
            _, loss_batch, acc_batch, train_summary = sess.run([optimize, loss, accuracy, merged], feed_dict={X: X_batch, Y: Y_batch})
            train_writer.add_summary(train_summary, train_steps)
            print('Average loss/acc batch {}: {} / {:.2f}%'.format(
                train_steps, loss_batch, acc_batch / len(X_batch) * 100))

    print('Total time: {0} seconds'.format(time.time() - start_time))

    print('Optimization Finished!') # should be around 0.35 after 25 epochs

    # test the model

    test_steps = 0
    total_correct_preds = 0

    for i in range(0, len(X_test), batch_size):
        test_steps += 1
        X_batch, Y_batch = X_test[i : i + batch_size], Y_test[i : i + batch_size]
        accuracy_batch, test_summary = sess.run([accuracy, merged], feed_dict={X: X_batch, Y:Y_batch})
        test_writer.add_summary(test_summary, test_steps)
        total_correct_preds += accuracy_batch

    print('Accuracy {:.2f}%'.format(total_correct_preds / len(X_test) * 100))
