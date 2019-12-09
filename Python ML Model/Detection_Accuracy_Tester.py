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

from time import time
import os
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Activation, Dropout, BatchNormalization
from keras import optimizers
import keras
from plot_confusion_matrix import pcm

"""
Tasks:
    1. Create/tweak model
    2. Perform Cross Validation and obtain accuracy
""" 

start = time()

try: 
    os.chdir(sys.argv[1])
except FileNotFoundError:
    sys.exit('Provided File directory not found. Exiting program.')

print('Loading data...\n')
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
    
    model_name = 'binary_1'
    
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=256, input_shape=(150,30,12), kernel_size=(15,3), strides=(5,1), padding='valid'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.4))
    model.add(Conv2D(filters=256, kernel_size=(4,4), strides=(1,1)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.4))
    model.add(Dense(1000, activation = 'sigmoid'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation = 'softmax'))
    #    model.add(Activation('softmax'))
#    adam = optimizers.Adam(lr = 0.005)
    sgd = keras.optimizers.SGD(lr = 0.1, momentum = 0.0, nesterov = False)
    
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd, metrics=['accuracy'])
    # *** END MODEL
    
    class_weight = {"Falling": 0.727, "Daily activities": 0.273} # Change as needed
    metric = model.fit(x=train_x, y=train_y, validation_data=(val_x,val_y), verbose=1, epochs=1, class_weights=class_weight).history
    print('Training Accuracy: ' + str(metric['acc']))
    print('Validation Accuracy: ' + str(metric['val_acc']))
    metrics[i+1] = (metric['acc'], metric['val_acc'])
    ave_train_acc += (metric['val_acc'][0]/folds)
    ave_val_acc += (metric['val_acc'][0]/folds)
    # Do prediction, reformat predictions, plot confusion matrix, and save
    print('Predicting...')
    val_y_predictions = model.predict(val_x, verbose = 1)
    # Create confusion matrix, display, and save
    pcm(val_y, val_y_predictions, classes = ['Falling', 'Sitting', 'Walking', 'Laying', 'Standing'], title='Confusion Matrix for Fold ' + str(i+1), name='confusion_matrix - ' + model_name + '_fold' + str(i+1))
    pred_list = np.append(pred_list, val_y_predictions)
    val_list = np.append(val_list, val_y)
    del val_x, val_y, train_x, train_y

# Create overall confusion matrix and print overall accuracy
pcm(val_list, pred_list, classes = ['Falling', 'Daily Activity'], title='Confusion Matrix for All Folds', name='confusion_matrix_' + model_name + '_overall')
print('\nAverage training accuracy: ' + str(ave_train_acc))
print('Average validation accuracy: ' + str(ave_val_acc))

end = time()
print('Total Running Time: {:0.3f} seconds.'.format(end-start))