#python Detection_Model_Trainer.py "D:\University Files\WVU\Fall 2019\CS 481 - Capstone Implementation\testData"
# If using on different computer, change above directory to full file location of testData

"""
Created on Mon Dec  2 15:10:45 2019
Neural Network TRAINING AND EXPORTING code for WiFi-Based In-home Fall-detection Utility (WIFU)
Precursor to training - run after Detection_Data_Creator.py and Detection_Accuracy_Tester.py
*** CURRENT command line argument format:
python Detection_Model_Trainer.py <testData directory>
@author: Matthew Keaton
"""

import os
import sys
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, Activation, MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers

from plot_confusion_matrix import pcm

"""
Tasks:
    1. Train entire dataset
    2. export model
""" 

# Load in data
try: 
    os.chdir(sys.argv[1])
except FileNotFoundError:
    sys.exit('Provided File directory not found. Exiting program.')

print('Loading data...')
fall_data = np.load('fall_data.npy')
data_size = len(fall_data)
fall_labels = np.load('fall_labels.npy')

# Model Creation - input the model to be trained here (don't forget to call model.compile())

model_name = 'test_5'
    
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
model.add(Dense(5, activation = 'softmax'))
#    model.add(Activation('softmax'))
adam = optimizers.Adam(lr = 0.005)
#    sgd = keras.optimizers.SGD(lr = 0.01, momentum = 0.0, nesterov = False)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
#%%
# *** END MODEL

# Train dataset and export model
metric = model.fit(x=fall_data, y=fall_labels)
#print('Training Accuracy: ' + str(metric['acc']))
#print('Validation Accuracy: ' + str(metric['val_acc']))
#%%
#print('Saving model...\n')
#np.save(model_name, model)
print('Predicting...')
pred_y = model.predict(fall_data, verbose = 1)
# turning predicted weights into list of predicted labels (for confusion matrix)
pred_vals = np.zeros((pred_y.shape[0],1))
for j in range(0, pred_y.shape[0]):
    max_val = 0
    max_ind = 0
    for k in range(0,pred_y.shape[1]):
        if pred_y[j][k] > max_val:
            max_ind = k
            max_val = pred_y[j][k]
    pred_vals[j] = max_ind

# turning one-hot encoded ground truth into multiclass encoding (for confusion matrix)
uncat_y = np.zeros((fall_labels.shape[0],1))
for j in range(0, fall_labels.shape[0]):
    for k in range(0,fall_labels.shape[1]):
        if fall_labels[j][k] == 1:
            uncat_y[j] = k
        
            
pcm(uncat_y, pred_vals, classes = ['Falling', 'Sitting', 'Walking', 'Laying', 'Standing'], title='Confusion Matrix for All Data', name='confusion_matrix_' + model_name + '_overall')