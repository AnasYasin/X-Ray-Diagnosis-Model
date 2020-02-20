import pandas as pd
import cv2
import os

data = pd.read_json('anasFinal.json')
data = data.transpose()
print(data.shape)
name = ''
count = 0
h = c = o = t = dt = do = 0
for multi in range(8):
    for i in range(907):    
        found = False
        name = ''
        reg = pd.DataFrame(data['regions'][i])
        reg = reg.transpose()
        
        try:
            if(reg['region_attributes'][multi]['Hair'] == 'X'):
                found = True
                ftype = 'Hair'
        except:
            pass
        
        try:
            if(reg['region_attributes'][multi]['C'] == 'X'):
                found = True
                ftype = 'C'
        except:
            pass
        
        try:
            if(reg['region_attributes'][multi]['T'] == 'X'):
                found = True
                ftype = 'T'
        except:
            pass
        
        try:
            if(reg['region_attributes'][multi]['O'] == 'X'):
                found = True
                ftype = 'O'
        except:
            pass
        
        try:
            if(reg['region_attributes'][multi]['DT'] == 'X'):
                found = True
                ftype = 'DT'
        except:
            pass
        
        try:
            if(reg['region_attributes'][multi]['DO'] == 'X'):
                found = True
                ftype = 'DO'
        except:
            pass
        
        if(found):
            filename = data['filename'][i]
            x=reg['shape_attributes'][multi]['x']
            y=reg['shape_attributes'][multi]['y']
            h=reg['shape_attributes'][multi]['height']
            w=reg['shape_attributes'][multi]['width']

            filename = "1/" + filename
            try:
                img = cv2.imread(filename)
                #cv2.imshow('res', img)
                #cv2.waitKey(0)
                crop_img = img[y:y+h, x:x+w]
                lable = "2/"+ ftype + str(count)
                if(filename.find('png') != -1):
                    lable += ".png"
                elif(filename.find('jpg') != -1):
                    lable += ".jpg"
                elif(filename.find('jpeg') != -1):
                    lable += ".jpeg"

                a = cv2.imwrite(lable, crop_img)
                print(a)
                sleep(1100)

            except:
                print(filename, "Is not Readable")
            
            #print("Image :", i, ", Region :", multi, "__Extracted__")
            print("Name :",filename, "Region :", multi, "Type :", ftype)
            count+=1
            if(ftype == 'Hair'):
                h+=1
            elif(ftype == 'O'):
                o+=1
            elif(ftype == 'T'):
                t+=1
            elif(ftype == 'C'):
                c+=1
            elif(ftype == 'DT'):
                dt+=1
            elif(ftype == 'DO'):
                do+=1
            
            

print("Total images:", count)
print("Hair:", h)
print("O:", o)
print("T:", t)
print("C:", c)
print("DT:", dt)
print("DO:", do)


'''

reg = pd.DataFrame(data['regions'][15])
reg = reg.transpose()
print(reg['region_attributes'][1]['DT'])


#print(reg['shape_attributes'][0]['y'])

print(reg['shape_attributes'][1]['x'])
print(reg['shape_attributes'][1]['y'])
print(reg['shape_attributes'][1]['width'])
print(reg['shape_attributes'][1]['height'])
'''





