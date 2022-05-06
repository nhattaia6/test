# Part 1 - Building the CNN
#importing the Keras libraries and packages
# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPool2D
# from keras.layers import Flatten
# from keras.layers import Dense, Dropout
# import keras

import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# # Initialing the CNN
# classifier = Sequential()

# # Step 1 - Convolutio Layer 
# classifier.add(Convolution2D(32, (3,  3), input_shape = (64, 64, 3), activation = 'relu'))

# #step 2 - Pooling
# classifier.add(MaxPooling2D(pool_size =(2,2)))

# # Adding second convolution layer
# classifier.add(Convolution2D(32, (3,  3), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size =(2,2)))

# #Adding 3rd Concolution Layer
# classifier.add(Convolution2D(64, (3,  3), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size =(2,2)))


# #Step 3 - Flattening
# classifier.add(Flatten())

# #Step 4 - Full Connection
# classifier.add(Dense(256, activation = 'relu'))
# classifier.add(Dropout(0.5))
# classifier.add(Dense(26, activation = 'softmax'))

# #Compiling The CNN
# classifier.compile(
#               optimizer = optimizers.SGD(lr = 0.01),
#               loss = 'categorical_crossentropy',
#               metrics = ['accuracy'])


classifier = Sequential()
classifier.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
classifier.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
classifier.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
classifier.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
classifier.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

classifier.add(Flatten())
classifier.add(Dense(units=4096,activation="relu"))
classifier.add(Dense(units=4096,activation="relu"))
classifier.add(Dense(units=16, activation="softmax"))

from tensorflow.keras.optimizers import Adam
opt = Adam(lr=0.001)
classifier.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')


#Part 2 Fittting the CNN to the image
# from keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory(
#         'action_net_v1/train',
#         target_size=(64, 64),
#         batch_size=32,
#         class_mode='categorical')

# test_set = test_datagen.flow_from_directory(
#         'action_net_v1/test',
#         target_size=(64, 64),
#         batch_size=32,
#         class_mode='categorical')

# model = classifier.fit_generator(
#         training_set,
#         steps_per_epoch=800,
#         epochs=25,
#         validation_data = test_set,
#         validation_steps = 6500
#       )

trdata = ImageDataGenerator()
training_set = trdata.flow_from_directory(directory="action_net_v1/train",target_size=(224,224))
tsdata = ImageDataGenerator()
test_set = tsdata.flow_from_directory(directory="action_net_v1/test", target_size=(224,224))

hist = classifier.fit_generator(steps_per_epoch=100,generator=training_set, validation_data= test_set, validation_steps=2,epochs=2,callbacks=[checkpoint,early])


#Saving the model
import h5py
classifier.save('Trained_model.h5')

print(hist.history.keys())
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('accuracy.png')
# summarize history for loss

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
#plt.show()

from keras.preprocessing import image
img = image.load_img("pre.jpg",target_size=(224,224))
img = np.asarray(img)
#plt.imshow(img)
img = np.expand_dims(img, axis=0)
from keras.models import load_model
saved_model = load_model("vgg16_1.h5")
output = saved_model.predict(img)
print("RS:", output)
