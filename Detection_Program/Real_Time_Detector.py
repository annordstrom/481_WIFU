#saveCsi.mat
#matValid.txt

"""
Created on Sun Dec  8 15:29:02 2019
Neural Network Implementation - trained model to detect falls in real-time
to be called in datastream directory
@author: Matthew Keaton
"""

"""
Tasks
    1. Load model
    2. In repetition:
        - load current matrix
        - run model
        - send prediction to Raspberry Pi
"""

import os
import keras.models
from scipy import io
import numpy as np
from time import sleep
from time import time

# 1. Load model
model = keras.models.load_model('fall_detector.h5')

mat_valid = open("matValid.txt", "r+")
valid = mat_valid.read()
mat_valid.seek(0)

while valid == '0':
    valid = mat_valid.read()
    mat_valid.seek(0)
    sleep(0.5)
    print('Waiting for valid signal...')
mat_valid.seek(0)
mat_valid.write('0')
mat_valid.close()
valid = '0'

saveCsi = io.loadmat('saveCsi.mat')
csi = saveCsi['csi_data']

while True:
    
    start = time()
    print('Predicting...')
    # Predict one at a time
    for i in range(0, (len(csi)//50)-2):
        print(str(i))
        
        csi_new = csi[i*50:i*50+150,:].reshape(1,150,6,30)
        csi_split_complex = np.zeros((1,150,12,30))
        for j in range(0,6):
            csi_split_complex[0,:,2*j,:] = csi_new[0,:,j,:].real
            csi_split_complex[0,:,2*j+1,:] = csi_new[0,:,j,:].imag    
        csi_split_complex = np.swapaxes(csi_split_complex, 2, 3)
        result = model.predict(csi_split_complex)
        if result[0][0] == 1:
            os.system("scp /home/wifu/host.txt pi@192.168.1.10")
            print('Fall Detected!')
        
    # Signal end of predictions
#    matValid = open("matValid.txt", "w+")
#    matValid.write('0')
#    matValid.close()
#    valid = '0'
    
    end = time()
    print('Total prediction time: {:.2f} seconds.'.format(end-start))
    
    matValid = open("matValid.txt", "r+")
    valid = matValid.read(1)
    matValid.close()
    
    while valid == '0': # wait for signal to indicate data is valid
        print('Waiting for valid signal...')
        sleep(0.5)
        matValid = open("matValid.txt", "r+")
        valid = matValid.read(1)
        matValid.close()
    
    # Once valid, read in data for next loop
    saveCsi = io.loadmat('saveCsi.mat')
    csi = saveCsi['csi_data']