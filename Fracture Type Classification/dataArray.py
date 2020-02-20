import cv2
import numpy as np
from os import walk
label_dict = {
    0: 'T',
    1: 'O',
    2: 'C',
    3: 'DT',
    4: 'DO',    
}
num_training = 209
img_size = 96

X = np.zeros((num_training, 1, img_size, img_size), dtype = np.float32)
y = np.zeros((num_training), dtype = np.int32)

mypath = '3/'
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break

for i in range(num_training):
    filename = "3/" + f[i]
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    X[i] = img

for i in range(num_training):
    if(f[i].find('DT') != -1):
        y[i]=3
    elif(f[i].find('DO') != -1):
        y[i]=4
    elif(f[i].find('T') != -1):
        y[i]=0
    elif(f[i].find('O') != -1):
        y[i]=1
    elif(f[i].find('C') != -1):
        y[i]=2

np.save('X_train.npy', X)
np.save('y_train.npy', y)

