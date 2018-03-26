#!/usr/bin/python
# simple mnist experiment

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import variational_dropout_tf as vd
from keras.datasets import fashion_mnist
from batch_iterator import iterate_minibatches


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def deepnn(x, phase):
    """
    Builds the network graph.

    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is the
        number of pixels in a standard MNIST image.
        x: True is train, False is test

    Returns:
        A tuple (y, log_alphas). y is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the digit into one of 10 classes (the
        digits 0-9). log_alphas is a list of the log_alpha parameters describing
        the effective dropout rate of the approximate posterior.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.variable_scope('conv1'):
        h_conv1 = vd.conv2d(x_image, phase, 32, [3,3])

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.variable_scope('conv2'):
        h_conv2 = vd.conv2d(h_pool1, phase, 64, [3,3])

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.variable_scope('fc1'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = vd.fully_connected(h_pool2_flat, phase, 128)
    with tf.variable_scope('fc2'):
        h_fc2 = vd.fully_connected(h_fc1, phase, 128)
    # Map the 1024 features to 10 classes, one for each digit
    with tf.variable_scope('fc2'):
        y_conv = vd.fully_connected(h_fc2, phase, 10)
    return y_conv

def main():
    # Import data
    #mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.reshape(x_train, (-1,784)) / 255.
    x_test = np.reshape(x_test, (-1, 784))  / 255.
    train_iterator = iterate_minibatches(x_train,y_train,batchsize=128,shuffle=True)
    test_iterator = iterate_minibatches(x_test, y_test, batchsize=128, shuffle=True)

    # Define placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.int32, [None,])

    y_onehot = tf.one_hot(y_, 10)
    phase = tf.placeholder(tf.bool, None)

    # Build the graph for the deep net
    with tf.name_scope('net1'):
        y_conv = deepnn(x, phase)

    gradients = tf.gradients(y_conv, xs=tf.trainable_variables())
    with tf.name_scope('loss'):
        # cross entropy part of the ELBO
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot,
                                                            logits=y_conv))
        # prior DKL part of the ELBO
        log_alphas = vd.gather_logalphas(tf.get_default_graph())
        divergences = [vd.dkl_qp(la) for la in log_alphas]
        # combine to form the ELBO

        #N = float(mnist.train.images.shape[0])
        N = x_train.shape[0]

        dkl = tf.reduce_sum(tf.stack(divergences))
        elbo = cross_entropy+(1./N)*dkl

    with tf.name_scope('adam_optimizer'):
         train_step = tf.train.AdamOptimizer(1e-3).minimize(elbo)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_onehot, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        
    with tf.name_scope('sparseness'):
        sparse = vd.sparseness(log_alphas)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            train_x, train_y = next(train_iterator)
            if i % 1000 == 0:
                train_accuracy, train_loss = sess.run((accuracy, elbo),
                    feed_dict={x: train_x, y_: train_y, phase: False})
                print('step %d, training accuracy %g, training loss: %g' %
                    (i, train_accuracy, train_loss))
                #val_x, val_y = mnist.validation.next_batch(50)
                val_x, val_y = next(test_iterator)
                val_accuracy, val_loss, val_sp = sess.run((accuracy, cross_entropy, sparse),
                    feed_dict={x: val_x, y_: val_y, phase: False})
                print('step %d, val accuracy %g, val loss: %g, sparsity: %g' %
                    (i, val_accuracy, val_loss, val_sp))
            train_step.run(feed_dict={x: train_x, y_: train_y, phase: True})
            # if i == 1000:
            #     w1 = sess.run(w)
            #     np.savetxt("W1.csv", w1, delimiter=",")
            # if i == 5000:
            #     w5 = sess.run(w)
            #     np.savetxt("W5.csv", w5, delimiter=",")
            # if i == 10000:
            #     w10 = sess.run(w)
            #     np.savetxt("W10.csv", w10, delimiter=",")
            # if i == 19000:
            #     w19 = sess.run(w)
            #     np.savetxt("W19.csv", w19, delimiter=",")
            # if i == 39000:
            #     w39 = sess.run(w)
            #     np.savetxt("W39.csv", w39, delimiter=",")
            # if i == 49000:
            #     w49 = sess.run(w)
            #     np.savetxt("W49.csv", w49, delimiter=",")
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: x_test, y_:y_test, phase: False}))


if __name__ == '__main__':
    main()
