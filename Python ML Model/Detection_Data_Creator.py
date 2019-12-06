#python Detection_Data_Creator.py "D:\University Files\WVU\Fall 2019\CS 481 - Capstone Implementation\testData"
# If using on different computer, change above directory to full file location of testData
"""
Created on Mon Nov 18 10:37:22 2019
Neural Network DATA COLLECTION code for WiFi-Based In-home Fall-detection Utility (WIFU)
Precursor to training - run prior to Detection_Model_Trainer.py
*** CURRENT command line argument format:
python Detection_Data_Creator.py <testData directory>
@author: Matthew Keaton
"""

import os
import sys
from scipy import io
import numpy as np
from collections import Counter
import random
from keras.utils import to_categorical

"""
Tasks:
    1. Load in training data and labels
    2. Shuffle data
    3. Perform one-hot encoding
    4. Normalize data
    5. Save training data and labels
""" 

#1.) Loading in training data and labels
try: 
    os.chdir(sys.argv[1])
except FileNotFoundError:
    sys.exit('Provided File directory not found. Exiting program.')

# Count number of samples first in order to allocate data
print('\nCounting the number of samples...')
data_count = 0
for file_dir in os.listdir(os.getcwd()): # rotate through each subject folder
    if os.path.isdir(os.path.join(os.getcwd(), file_dir)):
        folder_count = 0
        dirpath = os.path.join(os.getcwd(), file_dir)
        print(dirpath)
        for file in os.listdir(dirpath): # rotate through each .mat file in each subject folder
            if file.endswith('.mat') and os.path.isfile(os.path.join(dirpath, (file[:-4] + '.dat')))and os.path.isfile(os.path.join(dirpath, (file[:-4] + '.txt'))):
                filepath = os.path.join(os.getcwd(), file_dir, file)
                mat_in = io.loadmat(filepath)
                csi = np.asarray(mat_in['csi_data'])
                if csi.shape == (750,2,3,30):
                    folder_count += 1
            elif file.endswith('.mat'):
                print('Corrupt file ' + file + ': .dat or .txt file does not exist for this .mat file.')
        print('Test count: ' + str(folder_count))
        data_count += folder_count
print('Total test count: ' + str(data_count) + '\n') 

# Create proper sized data arrays
fall_data = np.zeros((data_count * 13,150,12,30),int)
fall_labels = np.zeros((data_count * 13),int)

# Collect CSI data and labels
print('Collecting data...')
i=0
for file_dir in os.listdir(os.getcwd()): # rotate through each subject folder
    if os.path.isdir(os.path.join(os.getcwd(), file_dir)):
        dirpath = os.path.join(os.getcwd(), file_dir)
        print(dirpath)
        for file in os.listdir(dirpath): # rotate through each .mat file in each subject folder
            if file.endswith('.mat') and os.path.isfile(os.path.join(dirpath, (file[:-4] + '.dat')))and os.path.isfile(os.path.join(dirpath, (file[:-4] + '.txt'))):
                try:
                    # LOAD CSI DATA
                    filepath = os.path.join(os.getcwd(), file_dir, file)
                    mat_in = io.loadmat(filepath)
                    csi = np.asarray(mat_in['csi_data'])
                    first_ind = 0
                    end_ind = 150
                    for j in range(0,13):
                        csi_new = csi[first_ind:end_ind]
                        csi_new = csi_new.reshape(1,150,6,30)
                        csi_split_complex = np.zeros((1,150,12,30))
                        for k in range(0,6):
                            csi_split_complex[0,:,2*k,:] = csi_new[0,:,k,:].real
                            csi_split_complex[0,:,2*k+1,:] = csi_new[0,:,k,:].imag
                        fall_data[13*i+j] = csi_split_complex
                        first_ind += 50
                        end_ind += 50
                    
                    
                    # LOAD LABELS
                    txt_file = os.path.join(dirpath, (file[:-4] + '.txt'))
                    with open(txt_file) as txt: # Create list from current file
                        file_labels = []
                        for line in txt:
                            file_labels.append(int(line[0]))
                            
                    # Determine which label to use for the current sample, based on majority
                    first_ind = 0
                    end_ind = 150
                    for j in range(0,13):
                        # Count each label
                        occurence_count = Counter(file_labels[first_ind:end_ind])
                        fall_labels[13*i + j] = occurence_count.most_common(1)[0][0]
                        first_ind += 50
                        end_ind += 50
                    
                    # Determine which labels to change to fall labels, based on percentage
                    label_ind = 0
                    first_one = 0
                    first_not_one = 0
                    while label_ind < len(file_labels): # Find first '1' - representing fall
                        if file_labels[label_ind] == 1:
                            first_one = label_ind
                            while label_ind < len(file_labels) and (file_labels[label_ind] == 1): # Find first non-1 - representing end of fall
                                label_ind += 1
                            first_not_one = label_ind
                            break
                        label_ind += 1
                    # If there is a fall, and 75% of fall is located in sample timeframe, relabel sample as fall
                    if first_not_one > 0:
                        fall_length = first_not_one - first_one
                        first_ind = 0
                        end_ind = 150
                        for j in range(0,13): # Look at each 3-second block, one at a time
                            if first_one < end_ind and first_not_one > first_ind: # If at least part of the fall 'falls' in the range of the 3-second segment
                                fall_captured = min(end_ind, first_not_one) - max(first_ind, first_one) # Calculate length of fall captured in 3-second segment
                                if fall_captured >= fall_length*.75 and fall_labels[13*i + j] != 1:
                                    fall_labels[13*i + j] = 1
                            first_ind += 50
                            end_ind += 50
                    
                    i+=1
                except ValueError:
                    print('Error on ' + file + ': Wrong array size. Data unusable. Size: ' + str(csi.shape))
            elif file.endswith('.mat'):
                print('Corrupt file ' + file + ': .dat or .txt file does not exist for this .mat file.')

# 2.) & 3.) Randomize data, swap axes and one-hot encode labels
fall_data = np.swapaxes(fall_data, 2, 3)
print('\nRandomizing Data...')
shuffled_fall_data = np.zeros((len(fall_data), 150, 30, 12), dtype=float)
shuffled_fall_labels = np.zeros(len(fall_labels))
shuffled_indices = list(range(len(fall_data)))
random.shuffle(shuffled_indices)
i = 0
for j in shuffled_indices:
    shuffled_fall_data[i] = fall_data[j]
    shuffled_fall_labels[i] = fall_labels[j]
    i += 1
shuffled_fall_labels = shuffled_fall_labels - 1
shuffled_fall_labels = to_categorical(shuffled_fall_labels)
del fall_data, fall_labels

# 4.) Normalize data
print('\nPart 1 of Normalization...')
mini = np.amin(shuffled_fall_data)
maxi = np.amax(shuffled_fall_data)

print('Part 2 of Normalization...')

#shuffled_fall_data = (shuffled_fall_data - mini)/(maxi - mini)
# The below option is very time-consuming, so avoid if you have enough RAM to use above call
for a in range(0,shuffled_fall_data.shape[0]):
    for b in range(0,shuffled_fall_data.shape[1]):
        for c in range(0,shuffled_fall_data.shape[2]):
            for d in range(0,shuffled_fall_data.shape[3]):
                shuffled_fall_data[a][b][c][d] = (shuffled_fall_data[a][b][c][d] - mini)/(maxi - mini)

# 5.) Save data
print('\n Saving data and labels...')
np.save('fall_data', shuffled_fall_data)
np.save('fall_labels', shuffled_fall_labels)
print('\nCompleted.')