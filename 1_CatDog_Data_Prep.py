# -*- coding: utf-8 -*-
!clear
import numpy as np
import pandas as pd
import os
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, MaxPooling2D, Dense
import keras

batch_size = 128
epochs = 100

TRAIN_DIR = "/home/turan/.config/spyder-py3/BTK_Keras/train/"
os.chdir(TRAIN_DIR)
filenames = os.listdir(TRAIN_DIR)
d = 0
c = 0
labels = []
features = []
counter = 0
for filename in filenames:
    if filename != 'cat_dog_dataset.csv':
        name = filename.split('.')[0]
        if name == 'dog':
            labels.append('dog')
            d = d + 1
        else:
            labels.append('cat')
            c = c + 1
        img = cv2.resize(cv2.imread(filename, 
                                    cv2.IMREAD_GRAYSCALE),
                         (50,50))
        print("Processing images...", counter)
        counter = counter + 1
        img.reshape(50,50)
        features.append([np.array(img)])
    else:
        continue
    
    

    
        
df = pd.DataFrame({'filenames': filenames, 'labels': labels})

labels = df.iloc[:,1:2].values
le = preprocessing.LabelEncoder()
labels[:,0] = le.fit_transform(labels)
ohe = preprocessing.OneHotEncoder()
labels = ohe.fit_transform(labels).toarray()
labels = pd.DataFrame(labels)

labels = labels.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(features, 
                                                    labels,
                                                    test_size = 0.33
                                                    )
x_test = np.array(x_test)
x_train = np.array(x_train)

x_train = x_train.reshape(-1, 50, 50, 1)
x_test = x_test.reshape(-1, 50, 50, 1)

'''
x_train = np.moveaxis(x_train, 1, -1)
x_test = np.moveaxis(x_test, 1, -1)
'''

y_train = y_train
y_test = y_test

print("Training process is started...")

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu', input_shape=(50,50,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics = ['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))




preds = np.argmax(model.predict(x_test))


















