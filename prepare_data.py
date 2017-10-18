# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:25:34 2017

@author: C
"""
from sklearn.cross_validation import train_test_split
from numpy import corrcoef
from pandas import DataFrame


class predict():
    def __init__(self, data, target):
        data = DataFrame(data)
        target = DataFrame(target)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, target, train_size=0.8, random_state=43
        )
    
    def show_raport(self):
        print(corrcoef(self.X_train.T))
        self.X_train.boxplot()
        
        
from sklearn.datasets import load_iris
data = load_iris()
iris_pr = predict(data.data, data.target)
iris_pr.show_raport()