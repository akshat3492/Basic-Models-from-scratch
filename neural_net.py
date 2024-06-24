"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    I trained the network with a MSE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a MSE loss. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        
        #adam parameters
        self.t = 0
        self.m = {}
        self.vk = {}

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix -> prev_layer dimension, current_layer dimension
            X: the input data -> 256 * 256
            b: the bias -> current_layer dimension N,
        Returns:
            the output
        """
        
        return np.dot(X, W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        
        return X * (X>0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        
        return 1 * (X>0)

    

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (y-p)**2/len(y)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        #print(self.params)
        self.outputs = {}
        self.outputs['A0'] = X #32, 512
        '''
        if num_layers = 3
        W1 = 512,256; Z1 = 32, 256; A1 = 32, 256
        W2 = 256,256; Z2 = 32,256; A2 = 32, 256
        W3 = 256,3; Z3 = 32, 3
        '''
        
        for i in range(1, self.num_layers+1):
            self.outputs['Z'+str(i)] = self.linear(self.params["W" + str(i)], self.outputs['A' + str(i-1)], self.params["b" + str(i)])
            if i!=self.num_layers:
                self.outputs['A'+str(i)] = self.relu(self.outputs['Z'+str(i)])
        return self.outputs['Z'+str(self.num_layers)]

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        n = len(y)
        #preds (256,3)
        '''
        if num_layers = 3
        dZ3 = 32,3; Z3 = A2W3 + b3, A2 = 32, 256, W3 = 256,3
        Z2 = A1W2 + b2; A2 32,256; A2 = relu(Z2); A2->(32,256) Z2->32, 256
        '''
        
        preds = self.outputs['Z'+str(self.num_layers)]
        loss = self.mse(y, preds)
        self.gradients['dZ'+str(self.num_layers)] = 2*(preds-y)/n
        
        for i in range(self.num_layers, 0, -1):
            self.gradients['dW'+str(i)] = self.outputs['A'+str(i-1)].T @ self.gradients['dZ'+str(i)]
            
            self.gradients['db'+str(i)] = np.sum(self.gradients['dZ'+str(i)], axis=0)
            
            if i!=1:
                self.gradients['dA'+str(i-1)] = (self.gradients['dZ'+str(i)] @ self.params['W'+str(i)].T)
                self.gradients['dZ'+str(i-1)] = self.gradients['dA'+str(i-1)] * self.relu_grad(self.outputs['Z'+str(i-1)])
        
         
        return np.sum(loss)

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD"
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        if opt=="SGD":
            for i in range(1, self.num_layers + 1):
                self.params["W" + str(i)] -= lr * self.gradients['dW'+str(i)]
                self.params["b" + str(i)] -= lr * self.gradients['db'+str(i)]
        else:
            self.t += 1
            for key in self.params.keys():
                if key not in self.m and key not in self.vk:
                    self.m[key] = 0
                    self.vk[key] = 0
                self.adam(key, b1, b2, eps, lr)
                
    def adam(self,key, b1, b2, eps, lr):
        
        self.m[key] = b1*self.m[key] + (1-b1)*self.gradients['d' + key]
        self.vk[key] = b2*self.vk[key] + (1-b2)*(self.gradients['d' + key]**2)
        
        m_hat = self.m[key]/(1-b1**self.t)
        vk_hat = self.vk[key]/(1-b2**self.t)
        
        self.params[key] = self.params[key] - lr*m_hat/(np.sqrt(vk_hat)+eps)
        