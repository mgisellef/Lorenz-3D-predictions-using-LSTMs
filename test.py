#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:53:23 2021

@author: fernandez48
"""

import os
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split

def f(x):
    return x*x

n=100
x=np.linspace(0,1,n)
y=f(x)

# Scaled to work with Neural networks.
scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(y.reshape(1,-1)).reshape(-1,)

x_train, x_test, y_train, y_test = train_test_split(x, y_scaled, test_size=0.2, shuffle=False)


SimpleRNN_model=Sequential()
SimpleRNN_model.add(SimpleRNN(128,input_shape=(1,1)))
SimpleRNN_model.add(Dense(1))
SimpleRNN_model.build()
print(SimpleRNN_model.summary())                    
