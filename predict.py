import tensorflow as tf
import sklearn
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from skimage import io, color
import pandas as pd
import preprocess as pre
import sys


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
    'wc4': tf.get_variable('W3', shape=(3,3,128,256), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W4', shape=(16*16*256,256), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W5', shape=(256,num_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bc4': tf.get_variable('B3', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B4', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B5', shape=(num_classes), initializer=tf.contrib.layers.xavier_initializer()),
}

def conv2d(X, W, b, strides = 1, name='conv'):
    with tf.name_scope(name):
        X = tf.nn.conv2d(X, W, strides = [1, strides, strides, 1], padding='SAME')
        X = tf.nn.bias_add(X, b)
        act = tf.nn.relu(X)
        return act

def maxpool2d(X, k = 2, name = 'maxpool'):
    with tf.name_scope(name):
        mp = tf.nn.max_pool(X, ksize=[1, k, k, 1], strides = [1, k, k ,1], padding = 'SAME')
        return mp

def fcl(X, W, b, name='fcl'):
    with tf.name_scope(name):
        act = tf.add(tf.matmul(X, W), b)
        return act

gap_layer = np.empty([10,256])
def global_avg_pooling(X, name='GAP'):
    gap = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(X)#, data_format='channels_last')
    res = np.array(gap)
    #np.save('GAP.npy', temp) 
    #print('GAP layer exported')

    print('Con4 =', res.shape)
    print(res)
    return


def conv_net(X, weights, biases):  
    conv1 = conv2d(X, weights['wc1'], biases['bc1'], name = 'conv1')
    conv1 = maxpool2d(conv1, k=2, name = 'maxpooling')

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], name = 'conv2')
    conv2 = maxpool2d(conv2, k=2, name = 'maxpooling')

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], name = 'conv3')
    conv3 = maxpool2d(conv3, k=2, name = 'maxpooling')

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], name = 'conv4')
    conv4 = maxpool2d(conv4, k=2, name = 'maxpooling')

    global_avg_pooling(conv4)

    flatten = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]]) 

    fc1 = fcl(flatten, weights['wd1'], biases['bd1'], name = 'fc1')
    relu = tf.nn.relu(fc1)
    
    logits = fcl(relu, weights['out'], biases['out'], name = 'fc2')

    return logits

def load_neg_data():
    print("loading data")   
    X = np.load('testSet.npy')
    y = np.load('testSetLabels.npy')
    

    neg = (np.asarray(np.where(y[:,1]==1)))
    
    neg = neg.flatten()

    total_indices = np.empty(752, dtype=int)

    total_indices[0:752] = neg[0:752]

    #shuffle
    
    np.random.shuffle(total_indices)

    y = y[total_indices]
    X = X[total_indices]

    
    print("X  Shape",X.shape)
    print("y  Shape",y.shape)
    return X, y


def load_pos_data():
    print("loading data")   
    X = np.load('testSet.npy')
    y = np.load('testSetLabels.npy')

    pos = (np.asarray(np.where(y[:,0]==1)))
    
    pos = pos.flatten()

    total_indices = np.empty(752, dtype=int)

    total_indices[0:752] = pos

    #shuffle
    
    np.random.shuffle(total_indices)

    y = y[total_indices]
    X = X[total_indices]

    
    print("X  Shape",X.shape)
    print("y  Shape",y.shape)
    return X, y

def load_equal_split_test_data():
    print("loading data")   
    X = np.load('testSet.npy')
    y = np.load('testSetLabels.npy')
    
    pos = (np.asarray(np.where(y[:,0]==1)))
    neg = (np.asarray(np.where(y[:,1]==1)))
    pos = pos.flatten()
    neg = neg.flatten()

    total_indices = np.empty(1504, dtype=int)
    total_indices[0:752] = pos
    total_indices[752:1504] = neg[0:752]

    #shuffle
    np.random.shuffle(total_indices)

    y = y[total_indices]
    X = X[total_indices]

    return X, y

def load_test_data():
    X = np.load('devSet.npy')
    y = np.load('devSetLabels.npy')
    return X, y

def plot_images(X):
    '''
    numpy_horizontal_concat = np.concatenate((X[0], X[1]), axis=1)
    for i in range(5):
        numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, X[i+2]), axis=1)

    cv2.imshow('image',numpy_horizontal_concat)
    cv2.waitKey(0)
    #time.sleep(1000)
    '''

    fig=plt.figure(figsize=(10, 10))
    columns = 3
    rows = 3
    print(X[0].shape)
    

    for i in range(1, columns*rows +1):
        img = np.squeeze(X[i-1], axis=2)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    plt.show()

    return


def predict (): 

    if (sys.argv[1] == 'p'):
        test_X, test_y = load_pos_data()
    elif (sys.argv[1] == 'n'):
        test_X, test_y = load_neg_data()
    elif (sys.argv[1] == 'e'):
        test_X, test_y = load_equal_split_test_data()
    elif (sys.argv[1] == 'a'):
        test_X, test_y = load_test_data()
    else:
        test_X, test_y = load_equal_split_test_data()
    #   plot_images(test_X)    

    X = tf.placeholder('float', [None, 256, 256,1], name = 'X')
    y = tf.placeholder('float', [None, num_classes], name = 'labels')

    logits = conv_net(X, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))     
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        
        print("Loading the saved model")
        saver.restore(sess, "/home/anas/FYP/weights/weights.ckpt")
        # all data
        batch_size = 64
        total_test_acc = 0
        total_test_loss = 0
        total_batch = len(test_X) // batch_size
        for test_batch in range(total_batch):                    
            test_batch_X = test_X[test_batch*batch_size:min((test_batch+1)*batch_size,len(test_X))]
            test_batch_y = test_y[test_batch*batch_size:min((test_batch+1)*batch_size,len(test_y))]          
            
            test_acc,test_loss = sess.run([accuracy, cost], feed_dict={X: test_batch_X, y : test_batch_y})
            prediction = sess.run(tf.argmax(logits, 1), feed_dict={X: test_batch_X, y: test_batch_y})
            actual = sess.run(tf.argmax(y, 1), feed_dict={X: test_batch_X, y: test_batch_y})
        

            total_test_acc += test_acc
            total_test_loss += test_loss  

        total_test_acc /= total_batch 
        total_test_loss /= total_batch 
        satck = tf.stack([actual,prediction],axis=1)  
        print(sess.run(satck))
        print("Testing Accuracy:","{:.5f}".format(total_test_acc), ", Testing loss:","{:.5f}".format(total_test_loss))
    return
        
if __name__ == "__main__":
    predict()
        








    
