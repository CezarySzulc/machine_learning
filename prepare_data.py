# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:25:34 2017

@author: C
"""
from sklearn.cross_validation import train_test_split


class predict():
    def __init__(self, data, target):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, target, train_size=0.8, random_state=43
        )
    