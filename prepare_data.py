# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:25:34 2017

@author: C
"""
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from numpy import corrcoef
from pandas import DataFrame


class predict_clf():
    def __init__(self, data, target):
        """ split data for training and test values """
        data = DataFrame(data)
        target = DataFrame(target)
        self.multi_class = True if len(target[0].unique()) > 2 else False
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, target, train_size=0.8, random_state=43
        )
    
    def show_raport(self, X):
        """ show correlation in datasets and plot boxplot """
        print(corrcoef(X.T))
        X = DataFrame(X)
        X.boxplot()
        
    def pipeline_k_neighbours(self, n_neighbors=5):
        pca = PCA()
        clf = KNeighborsClassifier(n_neighbors)
        # pipeline with pca and kneighbours
        pipeline = Pipeline(steps=[('pca', pca), ('clf', clf)])
        pipeline.fit(self.X_train, self.y_train.values.ravel())
        return pipeline
        
    def pipeline_logistic_reg(self):
        if self.multi_class:
            clf = LogisticRegression(multi_class='ovr')
        else:
            clf = LogisticRegression()
        scaler = StandardScaler()
        pipeline = Pipeline(steps=[('scaler', scaler), ('clf', clf)])
        pipeline.fit(self.X_train, self.y_train.values.ravel())
        return pipeline
        
    def test_classification(self, clf):
        print(clf.score(self.X_test, self.y_test))

        
from sklearn.datasets import load_iris
data = load_iris()
iris_pr = predict_clf(data.data, data.target)
#iris_pr.show_raport(iris_pr.X_train)
#iris_pr.decrease_datasets()
clf = iris_pr.pipeline_k_neighbours()
iris_pr.test_classification(clf)
clf = iris_pr.pipeline_logistic_reg()
iris_pr.test_classification(clf)