# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:25:34 2017

@author: C
"""
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from numpy import corrcoef, arange
from pandas import DataFrame


class predict_clf():
    def __init__(self, data, target):
        """ split data for training and test values """
        data = DataFrame(data)
        target = DataFrame(target)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, target, train_size=0.8, random_state=43
        )
    
    def show_raport(self, X):
        """ show correlation in datasets and plot boxplot """
        print(corrcoef(X.T))
        X = DataFrame(X)
        X.boxplot()
        
    def pipeline_k_neighbours(self):
        pca = PCA()
        clf = KNeighborsClassifier(5)
        # pipeline with pca and kneighbours
        pipeline = Pipeline(steps=[('pca', pca), ('clf', clf)])
        #best_k_param = {'clf__n': arange(0, 2, 0.1)}
        #search_func = GridSearchCV(pipeline, best_k_param)
        pipeline.fit(self.X_train, self.y_train)
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