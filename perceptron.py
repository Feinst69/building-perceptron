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
    def __init__(self, weights, bias, learning_rate):
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
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

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
        return 0 if weighted_sum < 0 else 1
        #return sigmoid(weighted_sum)
        
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
        
        Returns
        -------
            int : Loss value is 0 if output matches target, else 1.
        """
        output = self.forward(inputs)
        step = self.learning_rate * (target - output)
        self.bias += step
        self.weights +=  step * inputs
        return 1-abs(output-target)

__all__ = ["Perceptron"]