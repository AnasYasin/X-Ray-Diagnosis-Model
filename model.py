import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import time
import preprocess as pre
import train as model
import os
import os.path
import matplotlib.pyplot as plt

img_size = 256
epoch = 550
batch_size = 64
learning_rate = 0.00001
num_classes = 2

label_dict = {
    0: 'positive',
    1: 'negative'
}

def loadData():

    #loading training data________________________________________________________________
    print("Loading training images:")
    if (os.path.exists('trainingSet.npy') and os.path.exists('trainingSetLabels.npy')):
        X = np.load('trainingSet.npy')
        y = np.load('trainingSetLabels.npy')
    else:
        train_images_paths = pd.read_csv("train_image_paths.csv", header=None)
        #shuffle the df
        train_images_paths = train_images_paths.sample(frac=1).reset_index(drop=True)

        num_training = train_images_paths.__len__()
        print(num_training)
        X = np.zeros((num_training,img_size,img_size,1), dtype = np.float32)
        y = np.zeros((num_training, num_classes), dtype = np.float32)
        
        for index, row in train_images_paths.iterrows():
            if(row[0].find("positive") != -1):
                y[index,0] = 1

            elif(row[0].find("negative") != -1):
                y[index,1] = 1 
            
            img = cv2.imread(row[0], 0)

            #resizing image
            size = (img_size, img_size)    

            img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)

            #high pass filter
            sharp=pre.high_pass_filter(img)

            #sobel
            sobel , smooth_sobel = pre.sobel(sharp)
            
            #normalization
            #sobel = sobel - np.max(sobel)
            
            #standardization
            sobel = (sobel - np.mean(sobel)) / np.sqrt(np.var(sobel))

            sobel=np.reshape(sobel, (img_size,img_size,1))
            X[index] = sobel

            print("Training img : ", index)
      

    if (os.path.exists('testSet.npy') and os.path.exists('testSetLabels.npy')):
        test_X = np.load('testSet.npy')
        test_y = np.load('testSetLabels.npy')
    else:
        test_X = X[35000: 36809]
        test_y = y[35000: 36809]
        X_ = X[0:35000]
        y_ = y[0:35000]

        np.save('trainingSet.npy', X_)
        np.save('trainingSetLabels.npy', y_)
        np.save('testSet.npy', test_X)
        np.save('testSetLabels.npy', test_y)
        print(test_X.shape)
        print(X_.shape)
        print(X_)
        print(test_X)



    #loading testing data________________________________________________________________
    print("Loading testing images:")
    if (os.path.exists('devSet.npy') and os.path.exists('devSetLabels.npy')):
        dev_X = np.load('devSet.npy')
        dev_y = np.load('devSetLabels.npy')
        X_ = X
        y_ = y
    else:
        test_images_paths = pd.read_csv("valid_image_paths.csv", header=None)
        #shuffle df

        test_images_paths = test_images_paths.sample(frac=1).reset_index(drop=True)

        num_dev_set = test_images_paths.__len__()

        dev_X = np.zeros((num_dev_set,img_size,img_size,1), dtype = np.float32)
        dev_y = np.zeros((num_dev_set, num_classes), dtype = np.float32)

        for index, row in test_images_paths.iterrows():
            
            if(row[0].find("positive") != -1):
                dev_y[index,0] = 1

            elif(row[0].find("negative") != -1):
                dev_y[index,1] = 1 
            
            img = cv2.imread(row[0], 0)

            #resizing image
            size = (img_size, img_size)    
            img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)

            #high pass filter
            sharp=pre.high_pass_filter(img)

            #sobel
            sobel , smooth_sobel = pre.sobel(sharp)
            
            #normalization
            #sobel = sobel / sobel.max()
            
            #standardization
            sobel = (sobel - np.mean(sobel)) / np.sqrt(np.var(sobel))
            sobel=np.reshape(sobel, (img_size,img_size,1))

            dev_X[index] = sobel

            print("Dev set img : ", index)
            
        np.save('devSet.npy', dev_X) 
        np.save('devSetLabels.npy', dev_y)
    return X_, y_, dev_X, dev_y, test_X, test_y

def plot_images(X, y):
    cv2.imshow('image',X[0])
    cv2.waitKey(500)
    
    '''    
    if index ==10:

        #numpy_horizontal_concat = np.concatenate((img, blur), axis=1)
        #numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, diff), axis=1)
        numpy_horizontal_concat = np.concatenate((img, sharp), axis=1)
        #numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, uns_grad), axis=1)
        numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, smooth_sobel), axis=1)
        numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, sobel), axis=1)        

        cv2.imshow('image',numpy_horizontal_concat)
        cv2.waitKey(0)
        time.sleep(1000)
    ''' 



if __name__ == "__main__":
    X, y, dev_X, dev_y, test_X, test_y = loadData()

    model.train(X, y, dev_X, dev_y, epoch, learning_rate, batch_size)
    
    
