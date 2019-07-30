import tensorflow as tf
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"

num_classes = 2

label_dict = {
    0: 'positive',
    1: 'negative'
}
    
weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(32*32*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W4', shape=(128,num_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(num_classes), initializer=tf.contrib.layers.xavier_initializer()),
}

def conv2d(X, W, b, strides = 1, name='conv'):
    X = tf.nn.conv2d(X, W, strides = [1, strides, strides, 1], padding='SAME')
    X = tf.nn.bias_add(X, b)
    act = tf.nn.relu(X)
    return act

def maxpool2d(X, k = 2, name = 'maxpool'):
    mp = tf.nn.max_pool(X, ksize=[1, k, k, 1], strides = [1, k, k ,1], padding = 'SAME')
    return mp

def fcl(X, W, b, name='fcl'):
    act = tf.add(tf.matmul(X, W), b)
    return act

def conv_net(X, weights, biases):  
    conv1 = conv2d(X, weights['wc1'], biases['bc1'], name = 'conv1')
    conv1 = maxpool2d(conv1, k=2, name = 'maxpooling')

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], name = 'conv2')
    conv2 = maxpool2d(conv2, k=2, name = 'maxpooling')

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], name = 'conv3')
    conv3 = maxpool2d(conv3, k=2, name = 'maxpooling')

    flatten = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]]) 

    fc1 = fcl(flatten, weights['wd1'], biases['bd1'], name = 'fc1')
    relu = tf.nn.relu(fc1)
    
    logits = fcl(relu, weights['out'], biases['out'], name = 'fc2')

    return logits

def predict (train_X, train_y, learning_rate = 0.001):
    
    X = tf.placeholder('float', [None, 256, 256,1], name = 'X')
    y = tf.placeholder('float', [None, num_classes], name = 'labels')

    logits = conv_net(X, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))     
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, "/tmp/weights.ckpt")

        batch_x = train_X
        batch_y = train_y 

        loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x, y: batch_y})
        
        print ("Accuracy = {:.5f}".format(acc), ", Loss = {:.6f}".format(loss))

        return
