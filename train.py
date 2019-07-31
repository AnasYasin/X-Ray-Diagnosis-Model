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
        #tensorboard
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def maxpool2d(X, k = 2, name = 'maxpool'):
    with tf.name_scope(name):
        mp = tf.nn.max_pool(X, ksize=[1, k, k, 1], strides = [1, k, k ,1], padding = 'SAME')
        return mp

def fcl(X, W, b, name='fcl'):
    with tf.name_scope(name):
        act = tf.add(tf.matmul(X, W), b)
        #act = tf.nn.relu(fc)
        #tensorboard
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

def conv_net(X, weights, biases):  
    conv1 = conv2d(X, weights['wc1'], biases['bc1'], name = 'conv1')
    conv1 = maxpool2d(conv1, k=2, name = 'maxpooling')

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], name = 'conv2')
    conv2 = maxpool2d(conv2, k=2, name = 'maxpooling')

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], name = 'conv3')
    conv3 = maxpool2d(conv3, k=2, name = 'maxpooling')

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], name = 'conv4')
    conv4 = maxpool2d(conv4, k=2, name = 'maxpooling')

    flatten = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]]) 

    fc1 = fcl(flatten, weights['wd1'], biases['bd1'], name = 'fc1')
    relu = tf.nn.relu(fc1)
    
    tf.summary.histogram("fc1/relu", relu) #tensorboard
    logits = fcl(relu, weights['out'], biases['out'], name = 'fc2')

    return logits

def train (train_X, train_y, test_X, test_y, epoch = 550, learning_rate = 0.0001, batch_size = 64):
    X = tf.placeholder('float', [None, 256, 256,1], name = 'X')
    tf.summary.image('input', X, 3)
    y = tf.placeholder('float', [None, num_classes], name = 'labels')

    logits = conv_net(X, weights, biases)

    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar("xent", cost)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)     
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init) 

        summary_writer = tf.summary.FileWriter('./Output/114layered', sess.graph)
        summary_writer.add_graph(sess.graph)
        try:
            for i in range(epoch):
                for batch in range(len(train_X)//batch_size):   
                    batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
                    batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    

                    opt = sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
                    loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x, y: batch_y})
                    
                    #writing summries on every iteration.
                    s = sess.run(summ, feed_dict={X: batch_x, y: batch_y})
                    summary_writer.add_summary(s, i)
                    
                    if (batch % 25 == 0):
                        print("epoch: " + str(i) + ", batch: " + str(batch))
                        print ("Training Accuracy = {:.5f}".format(acc), ", Training Loss = {:.6f}".format(loss))

                        #test accuracy            
                        total_test_acc = 0
                        total_test_loss = 0
                        total_batch = len(test_X) // batch_size
                        for test_batch in range(total_batch):                    
                            test_batch_X = test_X[test_batch*batch_size:min((test_batch+1)*batch_size,len(test_X))]
                            test_batch_y = test_y[test_batch*batch_size:min((test_batch+1)*batch_size,len(test_y))]          
                            
                            test_acc,test_loss = sess.run([ accuracy,cost], feed_dict={X: test_batch_X, y : test_batch_y})
                            
                            total_test_acc += test_acc
                            total_test_loss += test_loss  

                        total_test_acc /= total_batch 
                        total_test_loss /= total_batch 
                        print("Testing Accuracy:","{:.5f}".format(total_test_acc), ", Testing loss:","{:.5f}".format(total_test_loss))
                        print("_____________________________________________________________")
        except KeyboardInterrupt:                
            print("Exporting Weights")
            save_path = saver.save(sess, "/tmp/weights.ckpt")
            summary_writer.close()  

    return
