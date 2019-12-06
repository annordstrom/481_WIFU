#python AlexNet_Tester.py "D:\University Files\WVU\Fall 2019\CS 481 - Capstone Implementation\testData"
# If using on different computer, change above directory to full file location of testData
"""
Created on Mon Dec  2 22:06:49 2019
Neural Network TESTING code for WiFi-Based In-home Fall-detection Utility (WIFU)
Precursor to training - run after Detection_Data_Creator.py
*** CURRENT command line argument format:
python Detection_Model_Trainer.py <testData directory>
@author: Matth
"""

import os
import sys
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization

"""
Tasks:
    1. xxxx
    2. xxxx
    3. Create model - how to do this? Start with basic Sequential model
    4. Create CV code, and tweak params
    5. xxxx
    6. xxx
""" 

try: 
    os.chdir(sys.argv[1])
except FileNotFoundError:
    sys.exit('Provided File directory not found. Exiting program.')

print('Loading data...')
fall_data = np.load('fall_data.npy')
data_size = len(fall_data)
fall_labels = np.load('fall_labels.npy')


#%%
# Task 3 & 4: Create Model, Perform CV and Test
folds = 5
metrics = {}
ave_train_acc = 0.0
ave_val_acc = 0.0
for i in range(0, folds):
    print('Fold ' + str(i+1) + ':')
    val_x = fall_data[i*data_size//folds:(i+1)*data_size//folds]
    val_y = fall_labels[i*data_size//folds:(i+1)*data_size//folds]
    if i == 0:
        train_x = np.array(fall_data[data_size//folds:])
        train_y = np.array(fall_labels[data_size//folds:])
    else:
        train_x = np.concatenate((fall_data[0:i*data_size//folds], fall_data[(i+1)*data_size//folds:]))
        train_y = np.concatenate((fall_labels[0:i*data_size//folds], fall_labels[(i+1)*data_size//folds:]))
    #Fresh model generation
    print('Loading...')
    
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(150,30,12), kernel_size=(7,3), strides=(3,2), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(7,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,2), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    
#    # 4th Convolutional Layer
#    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
#    model.add(Activation('relu'))
    
#    # 5th Convolutional Layer
#    model.add(Conv2D(filters=256, kernel_size=(3,2), strides=(1,1), padding='valid'))
#    model.add(Activation('relu'))
#    # Max Pooling
#    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    
    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    
    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    
    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    
    # Output Layer
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    model.summary()
    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy']) 
    
    metric = model.fit(x=train_x, y=train_y, validation_data=(val_x,val_y), verbose=1, epochs=1).history
    print('Training Accuracy: ' + str(metric['acc']))
    print('Validation Accuracy: ' + str(metric['val_acc']) + '\n')
    metrics[i+1] = (metric['acc'], metric['val_acc'])
    ave_train_acc += (metric['val_acc'][0]/folds)
    ave_val_acc += (metric['val_acc'][0]/folds)
    del val_x, val_y, train_x, train_y
print('\nAverage training accuracy: ' + str(ave_train_acc))
print('Average validation accuracy: ' + str(ave_val_acc))