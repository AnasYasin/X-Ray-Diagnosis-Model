import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import  time
import preprocess as pre


train_images_paths = pd.read_csv("train_image_paths.csv")
train_lables = pd.read_csv("train_labeled_studies.csv")
test_images_paths = pd.read_csv("valid_image_paths.csv")
test_lables = pd.read_csv("valid_labeled_studies.csv")

num_training = train_images_paths.__len__()

X = np.zeros((num_training, 62500), dtype = np.float16)

#loading training data
for index, row in train_images_paths.iterrows():
    img = cv2.imread(row[0], 0)

    #resizing image
    size = (250, 250)    
    img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)

    #high pass filter
    sharp=pre.high_pass_filter(img)

    #sobel
    sobel , smooth_sobel = pre.sobel(sharp)
    
    #normalization
    sobel = sobel / sobel.max()


    flat = np.matrix(sobel.flatten())
    
    X[index, :] = flat    
    

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
    
