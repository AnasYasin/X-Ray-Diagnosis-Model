import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import  time
import preprocess as pre
import train as tr
import os.path

label_dict = {
    0: 'positive',
    1: 'negative'
}
img_size = 250
def loadData():
    if (os.path.exists('trainingData.npy') and os.path.exists('trainingDataLabels.npy')):
        X = np.load('trainingData.npy')
        y = np.load('trainingDataLabels.npy')
    else:

        train_images_paths = pd.read_csv("train_image_paths.csv", header=None)
        #train_lables = pd.read_csv("train_labeled_studies.csv", header=None)
        test_images_paths = pd.read_csv("valid_image_paths.csv", header=None)
        #test_lables = pd.read_csv("valid_labeled_studies.csv", header=None)

        num_training = train_images_paths.__len__()
        num_classes = 2

        y = np.zeros((num_training, num_classes), dtype = np.float32)

        X = np.zeros((num_training,img_size,img_size), dtype = np.float16)

        #loading training data
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
            sobel = sobel / sobel.max()

            X[index] = sobel

            #flat = np.matrix(sobel.flatten())
            #X[index, :] = flat    

            #print(X.shape)
            print(index)
            
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
        np.save('trainingData.npy', X)
        np.save('trainingDataLabels.npy', y)


    return X, y

if __name__ == "__main__":
    X, y = loadData()
    print(X.shape, y.shape)
    print(X)
    print(y)
