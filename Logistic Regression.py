#!/usr/bin/env python
# coding: utf-8

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# from ucimlrepo import fetch_ucirepo 
# import random
  
# # fetch dataset 
# heart_disease = fetch_ucirepo(id=45) 
  
# # data (as pandas dataframes) 
# X = heart_disease.data.features 
# X = X.values

# y = heart_disease.data.targets 
# y = y.iloc[:, 0].values


# X, y = pd.DataFrame(X), pd.DataFrame(y)
# df = pd.concat([X,y], axis=1).dropna()


# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

import numpy as np
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# y = np.where(y!=0, 1, 0)
# y = np.where(y==0, -1, 1)



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression():
    def __init__(self, learning_rate, batch_size, num_of_iterations, threshold):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_of_iterations = num_of_iterations
        self.threshold = threshold
    
    def fit(self, X, y):
        self.X_train = X #N, D
        self.y_train = y #N,
        
        self.N = self.X_train.shape[0] #total number of training data points, N
        self.D = self.X_train.shape[1] #total number of dimensions, D
        
        self.w = np.random.rand(self.D, 1)# weights vector
        
    def train(self):
        for i in range(self.num_of_iterations):
            pred = np.dot(self.X_train, self.w) #(N,D) * (D,1) -> (N,1)
            margin = self.y_train.reshape(-1,1) * pred * (-1) #N,1
            h = sigmoid(margin) #(N,1)
            multiplication_part = self.X_train * self.y_train.reshape(-1,1) #N, D
            
            delta = np.dot(multiplication_part.T, h)
            self.w += self.learning_rate*delta
            
    def make_predictions(self, X_test):
        preds = np.dot(X_test, self.w)
        preds = sigmoid(preds)
        preds = np.where(preds>self.threshold, 1, -1)
        return preds
    
    def calculate_score (self, y_pred, y_test):
        return np.mean(y_test==y_pred.reshape(-1))



lr = LogisticRegression(learning_rate=0.01, batch_size=10, num_of_iterations=100, threshold=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lr.fit(X_train, y_train)

lr.train()

nn = lr.make_predictions(X_test)
lr.calculate_score(nn, y_test)

