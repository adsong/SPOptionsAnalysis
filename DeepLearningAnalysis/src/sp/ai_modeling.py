# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 10:51:11 2020

@author: Lee Sak Park
"""

from merge import Variable_selection
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import random



class AI():
    def __init__(self):
        pass
    
    def data_prep(self, t = 10, r = 20):
        self.import_data(t, r)
        self.scaling_data()
            
        
    def import_data(self, td = 10, rate = 20):
        raw = Variable_selection()
        raw.gen_data(td, rate)
        
        self.X = raw.X
        self.y = raw.y
        
        self.ANN_X = raw.ANN_X
        self.ANN_y = raw.ANN_y
        
        self.X_test = raw.X_test
        self.X_train = raw.X_train
        self.y_test = raw.y_test
        self.y_train = raw.y_train
        self.today = raw.today
        
        
        self.ANN_X_test = raw.ANN_X_test
        self.ANN_y_test = raw.ANN_y_test
        self.ANN_X_train = raw.ANN_X_train
        self.ANN_y_train = raw.ANN_y_train
        self.ANN_today = raw.ANN_today
        
        self.actual_diff = raw.actual_diff
        
        self.X = raw.X
        self.y = raw.y
        
        
    def scaling_data(self):
        
        ANN_X_test = self.ANN_X_test
        ANN_X_train = self.ANN_X_train
        ANN_today = self.ANN_today
        
        # scaling training/test data
        sc = StandardScaler()
        ANN_X_train = sc.fit_transform(ANN_X_train)
        ANN_X_test = sc.transform(ANN_X_test)
        ANN_today = sc.transform(ANN_today)

        
        # putting back the scaled results        
        self.ANN_X_train = ANN_X_train
        self.ANN_X_test = ANN_X_test
        self.ANN_today = ANN_today
        
        
        
    def test_ai(self, start_units = 15, end_units = 25):
        ANN_X_test = self.ANN_X_test
        ANN_X_train = self.ANN_X_train
        ANN_y_test = self.ANN_y_test
        ANN_y_train = self.ANN_y_train
        actual_diff = self.actual_diff
        
        rg = range(start_units, end_units)
        df = pd.DataFrame(actual_diff)
        
        
        
        
        for i in rg:
            classifier = Sequential()
    
            classifier.add(Dense(units = i, kernel_initializer = 'uniform', activation = 'relu', input_dim = 43))
            classifier.add(Dense(units = i+random.sample(range(1,10), 1)[0], kernel_initializer = 'normal', activation = 'relu'))
            classifier.add(Dense(units = i+random.sample(range(1,10), 1)[0], kernel_initializer = 'uniform', activation = 'relu'))
            classifier.add(Dense(units = i-random.sample(range(1,10), 1)[0], kernel_initializer = 'normal', activation = 'relu'))
            classifier.add(Dense(units = i, kernel_initializer = 'uniform', activation = 'relu'))
            classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
            classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            classifier.fit(ANN_X_train, ANN_y_train,batch_size = 20, epochs = 150)
            y_pred = classifier.predict(ANN_X_test)
            y_pred = y_pred.reshape((1, len(y_pred)))[0]
            df[str(i)] = y_pred.round(2)
        
        
        df['avg'] = df.iloc[:,1:].apply(np.mean, axis = 1)
        df['avg'] = df.avg.round(2)
        self.test_table = df
        
        
        
        
    def run_ai(self, n=21):
        ANN_today = self.ANN_today
        ANN_X = self.ANN_X
        ANN_y = self.ANN_y
        
        
        
        classifier = Sequential()


        classifier.add(Dense(units = n, kernel_initializer = 'uniform', activation = 'relu', input_dim = 43))
        classifier.add(Dense(units = n+random.sample(range(0,10), 1)[0], kernel_initializer = 'normal', activation = 'relu'))
        classifier.add(Dense(units = n+random.sample(range(0,10), 1)[0], kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = n-random.sample(range(0,5), 1)[0], kernel_initializer = 'normal', activation = 'relu'))
        classifier.add(Dense(units = n, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        classifier.fit(ANN_X, ANN_y,batch_size = 2, epochs = 250)
        y_pred = classifier.predict(ANN_today)
        y_pred = y_pred.reshape((1, len(y_pred)))[0]
        self.y_pred  = y_pred
        


    
        
        
        
        
        
        
        
        