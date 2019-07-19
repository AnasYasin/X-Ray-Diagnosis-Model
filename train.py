import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

iter = 550
learning_rate = 0.001
batch_size = 64
num_classes = 2

X = tf.placeholder('float', [None, 128, 128,1])
y = tf.placeholder('float', [None, num_classes])

label_dict = {
    0: 'positive',
    1: 'negative'
}

    
weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(16*16*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,num_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(num_classes), initializer=tf.contrib.layers.xavier_initializer()),
}


def conv2d(X, W, b, strides = 1, use_cudnn_on_gpu=True):

    X = tf.nn.conv2d(X, W, strides = [1, strides, strides, 1], padding='SAME')
    X = tf.nn.bias_add(X, b)
    return tf.nn.relu(X)

def maxpool2d(X, k = 2):
    mp = tf.nn.max_pool(X, ksize=[1, k, k, 1], strides = [1, k, k ,1], padding = 'SAME')
    return mp


def conv_net(X, weights, biases):  

    conv1 = conv2d(X, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)


    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


pred = conv_net(X, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)     

correct_prediction = tf.equal(tf.argmax(pred), tf.argmax(y))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

def train (train_X, train_y):
    with tf.Session() as sess:
        sess.run(init) 
        train_loss = []
        #test_loss = []
        train_accuracy = []
        #test_accuracy = []
        summary_writer = tf.summary.FileWriter('./Output', sess.graph)
        for i in range(iter):
            for batch in range(len(train_X)//batch_size):
                batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
                batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
                # Run optimization op (backprop).
                    # Calculate batch loss and accuracy
                opt = sess.run(optimizer, feed_dict={X: batch_x,
                                                                y: batch_y})

                loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x,
                                                                y: batch_y})
            print("Iter " + str(i) + ", Loss= " + \
                        "{:.6f}".format(loss) + ", Training Accuracy= " + \
                        "{:.5f}".format(acc))
            print("Optimization Finished!")
        summary_writer.close()
        return
