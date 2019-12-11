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
from time import time
from plot_confusion_matrix import pcm

"""
Tasks:
    1. Train entire dataset
    2. export model
""" 

start = time()

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
#%%
unique, counts = np.unique(fall_labels, return_counts=True)
fall_percent = counts[0]/len(fall_labels)
fall_weight = 1-fall_percent
daily_percent = counts[1]/len(fall_labels)
daily_weight = 1-daily_percent

class_weights = {0: fall_weight,
                 1: daily_weight} # Change as needed

# Model Creation - input the model to be trained here (don't forget to call model.compile())
model_name = 'binary_dense_3'
epoch = 10
    
model = Sequential()

model.add(Dense(12, input_shape=(150,30,12)))
model.add(Dense(6))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizers.Adam(), metrics=['accuracy'])
# *** END MODEL
#%%

# Train dataset and export model
model.summary()
print('Training...\n')
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

end = time()
print('Total Running Time: {:0.3f} seconds.'.format(end-start))