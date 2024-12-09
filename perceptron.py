""" Implementation of Perceptron
"""
import numpy as np 

class Perceptron:
    """ A class that implement a perceptron

    ...
        
        Attributes
        ----------
        weights (1D array/list of floats):
            The weights to apply.
        bias (float):
            The bias to apply.
        learning_rate (float):
            The rate to aplly while the model is learning.
    """
    def __init__(self):
        """
            The Perceptron constructor

            Parameters
            ----------
            weights (1D array/list of floats):
                The initial weights.
            bias (float):
                The initial bias.
            learning_rate (float):
                The rate to aplly while the model will be learning.
            
        """
        self.weights = None
        self.bias = 0.5
        self.learning_rate = 0.1
        self.predict = self.forward

    def activation(self, weighted_sum):
        """
        The activation function of the perceptron (name given by Fenitra doc)

        Arguments
        ---------
            weighted_sum (float):
                The result of the weighted summation.
                
        Returns
        -------
            int :  0 if the weighted summation is negative, else 1.
            
        """
        return np.where(weighted_sum < 0, 0, 1)

        
    def forward(self,inputs):
        """
        Computes the weighted summation pass it to the activation function.

        Arguments
        ---------
            inputs (array/list of floats):
                The inputs of the summation.

        Returns
        -------
            int or array of ints : 0 if the weighted summation is negative, else 1. 
            
        """
        weighted_sum = np.dot(inputs,self.weights) + self.bias
        return self.activation(weighted_sum)
        

    def learn(self, inputs, target):
        """
        Update weights and bias using the gradient descent method.

        Arguments
        ---------
            inputs (array/list of floats):
                The inputs we want to infer.
            target (int):
                The true value we would like to infer (0 or 1)

        """
        output = self.forward(inputs)
        step = self.learning_rate * (target - output)
        self.bias += step
        self.weights +=  step * inputs

    
    def fit(self,X,y, n_epochs=1000,lr=0.1):
        """
        Train the model.

        Arguments
        ---------
            X (array/list of floats):
                The inputs we want to infer.
            y (int):
                The true value we would like to infer (0 or 1)
            n_epochs (int):
                Number of training loops. Default is 1000.
            lr (float):
                Learning rate value. Default is 0.1.

        """
        if self.learning_rate is None:
            self.learning_rate = lr
        if self.weights is None:
            self.weights = np.random.random_sample(X.shape[1])
        for epoch in range(n_epochs):
            head = f"Epoch {epoch+1}/{n_epochs}"
            print(f"\r{head}", end="")
            for xi,yi in zip(X,y):
                self.learn(xi,yi)


__all__ = ["Perceptron"]