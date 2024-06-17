"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1/(1+np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
#         N, D = X_train.shape
        
#         self.w = np.zeros((1,D))
        
#         for epoch in range(self.epochs):
#             margins = -y_train.reshape(-1,1) * np.dot(X_train, self.w.T) #N*1 * (N,D @ D,1)
#             sigmoids = self.sigmoid(margins)
#             multi_term = y_train.reshape(-1,1) * X_train #N,D
            
#             self.w = self.w + self.lr*np.dot(sigmoids.T, multi_term)
#             self.lr = self.lr * 0.95
            
            
        
        N, D = X_train.shape
        
        self.w = np.zeros((1, D))
        
        for epoch in range(self.epochs):
            margins = np.dot(X_train, self.w.T) #N,D * D*1 -> N,1
            scores = self.sigmoid(margins) #N,1

            der_term = scores - y_train.reshape(-1,1) #N,1
            der_term = np.dot(der_term.T, X_train) #1,N * N*D -> 1*D

            self.w = self.w - self.lr * der_term
          
    
    
        
        
        
        #pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        preds = np.dot(X_test, self.w.T) #N,D * D,1 -> N,1
        preds = self.sigmoid(preds) #N,1
        preds = np.where(preds>self.threshold, 1, 0)
       
        return np.squeeze(preds)
