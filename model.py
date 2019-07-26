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
    if (os.path.exists('trainingData.npy') and os.path.exists('trainingDataLabels.npy')):
        X = np.load('trainingData.npy')
        y = np.load('trainingDataLabels.npy')
    else:
        train_images_paths = pd.read_csv("train_image_paths.csv", header=None)
        #shuffle the df
        train_images_paths = train_images_paths.sample(frac=1).reset_index(drop=True)

        num_training = train_images_paths.__len__()

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
            sobel = (sobel - np.mean(sobel)) / np.sqrt(np.var(sobel))

            sobel=np.reshape(sobel, (img_size,img_size,1))

            X[index] = sobel

            print("Training img : ", index)
            
        np.save('trainingData.npy', X)
        np.save('trainingDataLabels.npy', y)

    
    #loading testing data________________________________________________________________
    print("Loading testing images:")
    if (os.path.exists('testingData.npy') and os.path.exists('testingDataLabels.npy')):
        test_X = np.load('testingData.npy')
        test_y = np.load('testingDataLabels.npy')
    else:
        test_images_paths = pd.read_csv("valid_image_paths.csv", header=None)
        #shuffle df

        test_images_paths = test_images_paths.sample(frac=1).reset_index(drop=True)

        num_testing = test_images_paths.__len__()

        test_X = np.zeros((num_testing,img_size,img_size,1), dtype = np.float32)
        test_y = np.zeros((num_testing, num_classes), dtype = np.float32)

        for index, row in test_images_paths.iterrows():
            
            if(row[0].find("positive") != -1):
                test_y[index,0] = 1

            elif(row[0].find("negative") != -1):
                test_y[index,1] = 1 
            
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
            sobel = (sobel - np.mean(sobel)) / np.sqrt(np.var(sobel))


            sobel=np.reshape(sobel, (img_size,img_size,1))

            test_X[index] = sobel

            print("Testing img : ", index)
            
        np.save('testingData.npy', test_X)
        np.save('testingDataLabels.npy', test_y)
    return X, y, test_X, test_y

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
    X, y, test_X, test_y = loadData()

    
    model.train(X, y, test_X, test_y, epoch, learning_rate, batch_size)
    
    
