import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
classifier.load_weights('best_weights.hdf5')
test_image = image.load_img('D:\\University\\FYP\\Shapes\\image_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
print(len(classifier.layers))
result = classifier.predict(test_image)
#training_set.class_indices
print(result)
if (result[0][0] == 1):
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)    
layer_outputs = [layer.output for layer in classifier.layers[:5]]
print("working")
activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
print("working")
activations = activation_model.predict(test_image)
print("working")
first_layer_activation = activations[0]
print("working")
print(first_layer_activation.shape)
print("working")
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')