#python Detection_Accuracy_Tester.py "<filepath>/testData"
# If using on different computer, change above directory to full file location of testData

# Tip - To load in old saved models:
# new_model = keras.models.load_model('<model_path>/<model_file>')
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
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from keras import optimizers
import keras
from plot_confusion_matrix import pcm

"""
Tasks:
    1. Create/tweak model
    2. Perform Cross Validation and obtain accuracy
""" 

keras.backend.clear_session()

start = time()

try: 
    os.chdir(sys.argv[1])
except FileNotFoundError:
    sys.exit('Provided File directory not found. Exiting program.')

print('Loading data...\n')
fall_data = np.load('fall_data.npy')
data_size = len(fall_data)
fall_labels = np.load('fall_labels.npy')

unique, counts = np.unique(fall_labels, return_counts=True)
fall_percent = counts[0]/len(fall_labels)
fall_weight = 1-fall_percent
daily_percent = counts[1]/len(fall_labels)
daily_weight = 1-daily_percent
class_weights = {0: fall_weight,
                 1: fall_percent} # Change as needed

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
    model_name = 'Keatwood_14'
    epoch = 15
    
    model = Sequential()

    model.add(Dense(16, input_shape=(150,30,12)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Dropout(0.4)) # SHERWOOD 9 had 0.4
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizers.Adam(), metrics=['accuracy'])
    # *** END MODEL
    
    if i == 0:
        # Make directory for results
        cdir = sys.argv[1][:-8]
        cdir = cdir + '481_WIFU' + os.sep + 'Python ML Model' + os.sep + 'Confusion Matrices and Models' + os.sep + model_name + '_CM_M'
        os.mkdir(cdir)
        os.chdir(cdir)
        # Save initialized model before training (only needs to be done once)
        print('Saving model...')
        model.save('Testing_' + model_name)
        model.summary()
    
    metric = model.fit(x=train_x, y=train_y, validation_data=(val_x,val_y), verbose=1, epochs=epoch, class_weight=class_weights).history
    print('Training Accuracy: ' + str(metric['acc']))
    print('Validation Accuracy: ' + str(metric['val_acc']))
    metrics[i+1] = (metric['acc'], metric['val_acc'])
    ave_train_acc += (metric['val_acc'][0]/folds)
    ave_val_acc += (metric['val_acc'][0]/folds)
    # Do prediction, reformat predictions, plot confusion matrix, and save
    print('Predicting...')
    val_y_predictions = model.predict(val_x, verbose = 2)
    predicted_y = np.full(len(val_y_predictions), 2)
    for j in range(len(predicted_y)):
        if val_y_predictions[j] <= 0.5: # Falls
            predicted_y[j] = 0
        else: # Daily Activities
            predicted_y[j] = 1
    # Create confusion matrix, display, and save
    pcm(val_y, predicted_y, classes = ['Falling', 'Daily Activity'], title='Confusion Matrix for Fold ' + str(i+1), name='confusion_matrix - '+model_name+'_fold'+str(i+1), directory=cdir)
    pred_list = np.append(pred_list, predicted_y)
    val_list = np.append(val_list, val_y)
    del val_x, val_y, train_x, train_y

# Create overall confusion matrix and print overall accuracy

pcm(val_list, pred_list, classes = ['Falling', 'Daily Activity'], title='Confusion Matrix for All Folds', name='confusion_matrix_' + model_name + '_overall', directory=cdir)
print('\nAverage training accuracy: ' + str(ave_train_acc))
print('Average validation accuracy: ' + str(ave_val_acc))
# Below only works for binary classification: measurement for precision and recall
t_p = 0
t_n = 0
f_p = 0
f_n = 0
for i in range(0, len(val_list)):
    if val_list[i] == 0 and pred_list[i] == 0: # True Positive
        t_p += 1
    elif val_list[i] == 1 and pred_list[i] == 1: # True Negative
        t_n += 1
    elif val_list[i] == 1 and pred_list[i] == 0: # False Positive
        f_p += 1
    else: # val_list[i] == 0 and pred_list[i] == 1: False Negative
        f_n += 1
if t_p + f_p == 0:
    precision = 0
else:
    precision = t_p / (t_p + f_p)
if t_p + f_n == 0:
    recall = 0
else:
    recall = t_p / (t_p + f_n)
t = open(model_name + 'PvR.txt', 'w+')
t.write('Precision: ' + str(precision))
t.write('\nRecall: ' + str(recall))
t.close()
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
print('')
end = time()
print('Total Running Time: {:0.3f} seconds.'.format(end-start))
print('Model name: ' + model_name)