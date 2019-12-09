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
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras import optimizers

from plot_confusion_matrix import pcm

"""
Tasks:
    1. Train entire dataset
    2. export model
""" 

keras.backend.clear_session()

# Load in data
try: 
    os.chdir(sys.argv[1])
except FileNotFoundError:
    sys.exit('Provided File directory not found. Exiting program.')

print('Loading data...\n')
fall_data = np.load('fall_data.npy')
data_size = len(fall_data)
fall_labels = np.load('fall_labels.npy')

# Model Creation - input the model to be trained here (don't forget to call model.compile())
#%%
model_name = 'binary_conv_2'
epoch = 10
    
model = Sequential()
model.add(Conv2D(filters = 64, input_shape=(150,30,12), kernel_size = (13,4), strides = (4,2), padding = 'valid', activation = 'sigmoid'))
#    model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(filters = 32, kernel_size = (2,2), strides = (2,2), activation = 'sigmoid'))
#    model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizers.RMSprop(), metrics=['accuracy'])
# *** END MODEL
#%%

# Train dataset and export model
model.summary()
print('Training...\n')
class_weights = {0: 0.727,
                 1: 0.273} # Change as needed
metric = model.fit(x=fall_data, y=fall_labels, class_weight=class_weights, epochs=epoch)
print('Saving model...\n')
model.save(os.getcwd() + os.sep + 'fall_detector.h5')
#print('Training Accuracy: ' + str(metric['acc']))
#print('Validation Accuracy: ' + str(metric['val_acc']))
#%%
print('Predicting...')
pred_y = model.predict(fall_data, verbose = 1)
predicted_y = np.full(len(pred_y), 2)
for i in range(len(pred_y)):
    if pred_y[i] <= 0.5: # Falls
        predicted_y[i] = 0
    else: # Daily Activities
        predicted_y[i] = 1
pcm(fall_labels, predicted_y, classes = ['Falling', 'Daily Activity'], title='Confusion Matrix for All Data', name='confusion_matrix_' + model_name + '_trained')