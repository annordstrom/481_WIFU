#python Detection_Accuracy_Tester.py "D:\University Files\WVU\Fall 2019\CS 481 - Capstone Implementation\testData"
# If using on different computer, change above directory to full file location of testData
"""
Created on Mon Dec  2 22:06:49 2019
Neural Network TESTING code for WiFi-Based In-home Fall-detection Utility (WIFU)
Precursor to training - run after Detection_Data_Creator.py (may have to clear variables in between)
Use to create and optimize model before running Detection_Model_Trainer.py
*** CURRENT command line argument format:
python Detection_Model_Trainer.py <testData directory>
@author: Matth
"""

import os
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Activation, Dropout
import keras
from plot_confusion_matrix import pcm

"""
Tasks:
    1. Create/tweak model
    2. Perform Cross Validation and obtain accuracy
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
folds = 5
metrics = {}
ave_train_acc = 0.0
ave_val_acc = 0.0
pred_list = np.empty((0,1))
val_list = np.empty((0,1))
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
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    
    # *** END MODEL
    
    metric = model.fit(x=train_x, y=train_y, validation_data=(val_x,val_y), verbose=1, epochs=1).history
    print('Training Accuracy: ' + str(metric['acc']))
    print('Validation Accuracy: ' + str(metric['val_acc']))
    metrics[i+1] = (metric['acc'], metric['val_acc'])
    ave_train_acc += (metric['val_acc'][0]/folds)
    ave_val_acc += (metric['val_acc'][0]/folds)
    # Do prediction, reformat predictions, plot confusion matrix, and save
    print('Predicting...')
    pred_val_y = model.predict(val_x, verbose = 1)
    # turning predicted weights into list of predicted labels (for confusion matrix)
    pred_vals = np.zeros((pred_val_y.shape[0],1))
    for i in range(0,pred_val_y.shape[0]):
        max_val = 0
        max_ind = 0
        for j in range(0,pred_val_y.shape[1]):
            if pred_val_y[i][j] > max_val:
                max_ind = j
                max_val = pred_val_y[i][j]
        pred_vals[i] = max_ind
    # turning one-hot encoding into multiclass encoding (for confusion matrix)
    uncat_val_y = np.zeros((val_y.shape[0],1))
    for j in range(0, uncat_val_y.shape[0]):
        max_ind = 0
        for k in range(0,val_y.shape[1]):
            if val_y[j][k] == 1:
                max_ind = k
        uncat_val_y[j] = max_ind
    # Create confusion matrix, display, and save
    pcm(uncat_val_y, pred_vals, classes = ['Falling', 'Sitting', 'Walking', 'Laying', 'Standing'], title='Confusion Matrix for Fold ' + str(i+1), name='confusion_matrix' + str(i+1))
    pred_list = np.append(pred_list, pred_vals)
    val_list = np.append(val_list, uncat_val_y)
    del val_x, val_y, train_x, train_y

# Create overall confusion matrix and print overall accuracy
pcm(val_list, pred_list, classes = ['Falling', 'Sitting', 'Walking', 'Laying', 'Standing'], title='Confusion Matrix for Fold ' + str(i+1), name='confusion_matrix' + str(i+1))
print('\nAverage training accuracy: ' + str(ave_train_acc))
print('Average validation accuracy: ' + str(ave_val_acc))