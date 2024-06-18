"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        return

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        Initialize self.w as a matrix with random numbers in the range [0, 1)

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, D = X_train.shape
        self.w = np.random.rand(self.n_class, D) #C, D
        
        self.batch_size = 32
        n_batch = int(np.ceil(N/self.batch_size))
        for epoch in range(self.epochs):
            w_grad = np.zeros(self.w.shape) # (C,D)
            
            idx = np.random.permutation(N)
            X_train,y_train = X_train[idx], y_train[idx]
            
            for batch in range(n_batch):
                x = X_train[batch*self.batch_size:(1+batch)*self.batch_size] #n, D
                y = y_train[batch*self.batch_size:(1+batch)*self.batch_size] #n,
                
                scores = x @ self.w.T #n,C
                scores_yi = scores[np.arange(len(x)), y].reshape(-1,1) #n, 1
                
                I = 1 * (scores_yi - scores < 1) #n, C
                I[np.arange(len(x)), y] = 0
                
                #a far better vectorized approach, for this just look notes for formula to understand better
                #I[np.arange(len(x)), y] = -np.sum(I, axis=1)#n,C
                #dW = I.T @ x #C,n @ n,D
                #dW/=len(x)
                #dW += self.reg_const * self.w
                
                for ith in range(len(x)):
                    xi, yi = x[ith], y[ith]
                    w_grad[yi, :] -= np.sum(I[ith,:]) * xi
                
                w_grad[:,:] += np.dot(I.T, x) #C,N @ N, D -> C,D
                w_grad /= len(x)
                w_grad += self.reg_const * self.w
                
                #weight update
                self.w -= self.lr * w_grad  
            self.lr = self.lr * 0.95
 
        
        return

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
        preds = X_test @ self.w.T  #N, C
        
        return np.argmax(preds, axis=1)
