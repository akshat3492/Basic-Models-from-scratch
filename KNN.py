#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import preprocessing

#ensure to input the data as numpy arrays and not dataframe.

# Define KNN Class

def calculate_distance(test, train, method="euclidean"):
    if method == "manhattan":
        return np.sum(np.abs(test-train))
    return np.sqrt(np.sum((test - train)**2)) 

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)





class KNN:
    def __init__(self, k=3, metric="euclidean"):
        self.k = k
        self.metric = metric
        
    def fit(self, X, y, preprocess_=False):
        if preprocess_:
            X, y = pd.DataFrame(X), pd.DataFrame(y)
            df = pd.concat([X,y], axis=1).dropna()
            X = preprocessing.normalize(df.iloc[:, :-1].values)
            y = df.iloc[:, -1].values
            self.X_train = X
            self.y_train = y
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = [self.make_predictions(x) for x in X]
        return np.array(predictions)
    
    def make_predictions(self, x):
        distances = [calculate_distance(x, x_train, self.metric) for x_train in self.X_train]
        indices = np.argsort(distances)[:self.k]
        nearest_labels = [self.y_train[i] for i in indices]
        pred = np.bincount(nearest_labels).argmax()
        return pred





