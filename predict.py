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
from matplotlib import pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"]="0"

num_classes = 2

label_dict = {
    0: 'positive',
    1: 'negative'
}
    
weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc5': tf.get_variable('W4', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc7': tf.get_variable('W6', shape=(3,3,128,256), initializer=tf.contrib.layers.xavier_initializer()),
    'wc9': tf.get_variable('W8', shape=(3,3,256,512), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W9', shape=(8*8*512,512), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W10', shape=(512,num_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}

biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc5': tf.get_variable('B4', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bc7': tf.get_variable('B6', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bc9': tf.get_variable('B8', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B11', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B10', shape=(num_classes), initializer=tf.contrib.layers.xavier_initializer()),
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
    conv2 = maxpool2d(conv1, k=2, name = 'maxpooling')

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], name = 'conv3')
    conv4 = maxpool2d(conv3, k=2, name = 'maxpooling')
    
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], name = 'conv5')
    conv6 = maxpool2d(conv5, k=2, name = 'maxpooling')

    conv7 = conv2d(conv6, weights['wc7'], biases['bc7'], name = 'conv7')
    conv8 = maxpool2d(conv7, k=2, name = 'maxpooling')
    
    conv9 = conv2d(conv8, weights['wc9'], biases['bc9'], name = 'conv9')
    conv10 = maxpool2d(conv9, k=2, name = 'maxpooling')

    global_avg_pooling(conv10)

    flatten = tf.reshape(conv10, [-1, weights['wd1'].get_shape().as_list()[0]]) 

    fc1 = fcl(flatten, weights['wd1'], biases['bd1'], name = 'fc1')
    relu = tf.nn.relu(fc1)
    
    #tf.summary.histogram("fc1/relu", relu) #tensorboard
    logits = fcl(relu, weights['out'], biases['out'], name = 'fc2')

    return logits, conv10

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
    

    for i in range(1, columns*rows + 1):
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

    logits, GAP = conv_net(X, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))     
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        
        print("Loading the saved model")
        saver.restore(sess, "/home/anas/FYP/weights/newArch2.0/weights.ckpt")
        # all data
        batch_size = 64
        total_test_acc = 0
        total_test_loss = 0
        total_batch = len(test_X) // batch_size
        counter = 0
        for test_batch in range(total_batch):                    
            test_batch_X = test_X[test_batch*batch_size:min((test_batch+1)*batch_size, len(test_X))]
            test_batch_y = test_y[test_batch*batch_size:min((test_batch+1)*batch_size, len(test_y))]          
            
            test_acc, test_loss = sess.run([accuracy, cost], feed_dict={X: test_batch_X, y : test_batch_y})
            prediction = sess.run(tf.argmax(logits, 1), feed_dict={X: test_batch_X, y: test_batch_y})
            actual = sess.run(tf.argmax(y, 1), feed_dict={X: test_batch_X, y: test_batch_y})
            
            map = sess.run(GAP, feed_dict={X: test_batch_X, y : test_batch_y})
            print(total_batch)
            
            temp = map[0,:,:,0]
            dim=(256,256)
            temp = cv2.resize(temp, dim, interpolation = cv2.INTER_NEAREST)
            temp = cv2.GaussianBlur(temp, (11,11), 0)
 


            for i in range(total_batch):
                img = map[0,:,:,i+1]
                img = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
                img = cv2.GaussianBlur(img, (11,11), 0)
 
                temp += img


            fig=plt.figure(figsize=(10, 10))
            columns = 2
            rows = 1

            fig.add_subplot(rows, columns, 1)
            plt.imshow(test_batch_X[0,:,:,0], cmap='gray')

            fig.add_subplot(rows, columns, 2)
            plt.imshow(temp, cmap=plt.cm.jet, alpha=1)
            plt.colorbar()
            plt.show()
            
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
        








    
