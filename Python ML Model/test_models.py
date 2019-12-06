# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:23:10 2019

@author: Matth
"""

model = Sequential()
model.add(Conv2D(32, input_shape=(150, 30, 12), kernel_size=(3,3), strides=(3,3), activation='sigmoid'))
model.add(Conv2D(32, kernel_size=(25,6), strides=(1,1), activation='sigmoid'))
model.add(Flatten())
model.add(Dense(5, activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

