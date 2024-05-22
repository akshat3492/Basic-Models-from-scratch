#!/usr/bin/env python
# coding: utf-8


import numpy as np

# Sample Data
X = np.array([
    [1.1, 2.2, 3.3],
    [2.0, 3.0, 4.0],
    [3.1, 4.1, 5.1],
    [4.0, 5.0, 6.0],
    [5.1, 6.2, 7.3],
    [6.0, 7.0, 8.0],
    [7.1, 8.1, 9.1],
    [8.0, 9.0, 10.0],
    [9.1, 10.2, 11.3],
    [10.0, 11.0, 12.0]
])

y = np.array([10.1, 12.2, 14.1, 16.0, 18.3, 20.2, 22.3, 24.0, 26.1, 28.0])


class LinearRegression:
    
    def __init__(self,iterations, learning_rate, accuracy_metric="MSE"):
        self.accuracy_metric = accuracy_metric
        self.iterations = iterations
        self.learning_rate = learning_rate
        
    def fit(self, X, y):
        self.N = X.shape[0]
        self.w = np.zeros(X.shape[1]) #(f,)
        self.X_train = X
        self.y_train = y
        
        
    def train(self):
        for iters in range(self.iterations):
            y_pred = np.dot(self.X_train, self.w)#N,f * (f,) -> (N,)
            error = self.y_train - y_pred
            loss = (1/self.N) * np.sum(error**2)
            dw = -2*(np.dot(self.X_train.T, error)/self.N)
            self.w = self.w - self.learning_rate*dw
            print(loss)
    def make_predictions(self, X_test):
        return np.dot(X_test, self.w)
    
    def get_metric_score(self, preds, y_test):
        return np.mean((y_test-preds)**2)
        
            

linear_regressor = LinearRegression(iterations=10, learning_rate=0.001)
linear_regressor.fit(X, y)
linear_regressor.train()



