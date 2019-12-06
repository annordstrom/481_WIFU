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
#import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, Activation, MaxPooling2D
from keras.utils import to_categorical

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
fall_labels = fall_labels - 1
fall_labels_ohe = to_categorical(fall_labels)

# Model Creation - input the model to be trained here (don't forget to call model.compile())

model_name = 'base_model_1'

model = Sequential()
model.add(Conv2D(32, input_shape=(150, 30, 12), kernel_size=(3,3), strides=(3,3), activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# End of model creation

# Train dataset and export model
model.fit(x=fall_data, y=fall_labels_ohe)
np.save(model_name, model)