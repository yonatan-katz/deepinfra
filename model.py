#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:07:13 2017

@author: yonatank
"""
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from deepinfra import datasource
import numpy as np

    
def my_metrics(y_true, y_pred):
    return np.corrcoef(y_true,y_pred)[0][1]

def simple_sin(layer_width=64):
    model = Sequential()
    
    model.add(Dense(layer_width,input_shape=(1,),
              activation='sigmoid',
              kernel_initializer=initializers.random_normal(stddev=1/np.sqrt(layer_width))))
    
    model.add(Dense(layer_width,activation='sigmoid',
        kernel_initializer=initializers.random_normal(stddev=1/np.sqrt(layer_width))))
    
    model.add(Dense(layer_width,activation='sigmoid',
        kernel_initializer=initializers.random_normal(stddev=1/np.sqrt(layer_width))))    
    
    model.add(Dense(layer_width,activation='sigmoid',
        kernel_initializer=initializers.random_normal(stddev=1/np.sqrt(layer_width))))    
    
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(lr=0.1),
        metrics=[])
    
    #sin_train_generator = datasource.batch_sin_generator(batch_size=32)
    #model.fit_generator(sin_train_generator,steps_per_epoch=512,epochs=10)
    
    test_set_size = 1024
    test_x = np.empty(test_set_size)
    test_y = np.empty(test_set_size)
    index = 0
    for x,y in datasource.sin_function_generator(test_set_size):
        test_x[index] = x
        test_y[index] = y
        index +=1
        
    model.fit(x=test_x,y=test_y,batch_size=1024,epochs=1500)
        
    predict_y = model.predict(test_x,batch_size=1024)
    
    return predict_y,test_y
    
    
    
    
    
    