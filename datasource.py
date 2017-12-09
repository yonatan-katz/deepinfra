#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:11:27 2017

@author: yonatank
"""
import numpy as np
import pandas as pd

def sin_function_generator(samples):
    freq = 10
    for x in range(samples):
        y = np.sin(2*np.pi*freq*(x/samples)) + np.random.normal(0,0.2)
        yield x,y
        
def quadratic_function_generator(samples):    
    for x in range(samples):
        x = np.random.normal(scale=3)
        y = 0.2 * np.power(x,2) + np.random.normal(scale=0.1)
        yield x,y
        

def test_generator(generator=quadratic_function_generator):
    X = []
    Y = []
    for x,y in quadratic_function_generator(1000):
        X.append(x)
        Y.append(y)
    return pd.DataFrame({"x":X,"y":Y}) 
        
    

def batch_generator(batch_size,batch_num=None):
    x_batch = np.empty(batch_size)
    y_batch = np.empty(batch_size)
    
    batch_index = 0
    while True:    
        index = 0
        if batch_num is not None:
            if batch_index == batch_num:
                break
        batch_index += 1
        
        for x,y in quadratic_function_generator(batch_size):
            x_batch[index] = x
            y_batch[index] = y
            index += 1
        yield x_batch,y_batch
        
        


        
    
    
    
        
    
