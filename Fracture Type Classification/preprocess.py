import cv2
from os import walk

mypath = '2/'
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break

width = height = 0

for i in range(209):
    
    filename = "2/" + f[i]

    img = cv2.imread(filename)

    #resizing image
    size = (96, 96)    
    img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
    filename = "3/" + f[i]
    cv2.imwrite(filename, img)
    
print("w:",width/209)
print("h:", height/209)

    
