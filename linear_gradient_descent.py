# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:24:48 2024

Linear Regression Model Implementation using batch gradient descent -
used given load_data function to read and preprocess the dataset:
    - handling NaN values
    - extracting the features as input and 'mpg' as output for linear regression training.
    
The model consists of methods for initializing parameters, performing forward computation,
computing the cost (Mean Squared Error), computing gradients, updating parameters, making predictions,
and training the model.
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def initialize_parameters(self, num_features):
        """
        Initializes the parameters (weights and bias) of least sqaures linear regression model with zeros.
        (Sufficient because convex problem)

        Input:
        - num_features: Number of features in the input data

        Output:
        - None
        """
        self.b = np.zeros(1)
        self.W = np.zeros(num_features)
    
    def model_forward(self, X):
        """
        Performs forward computation for the linear regression model.

        Input:
        - X: Input data, shaped as (number of data points, number of features)

        Output:
        - predictions: Model predictions for the given input data
        """
        return (np.dot(X, self.W) + self.b).flatten()
    
    def compute_cost(self, y, predictions):
        """
        Computes the cost (Mean Squared Error) of the model.

        Input:
        - y: Actual target values
        - predictions: Model predictions

        Output:
        - cost: Mean Squared Error between predictions and actual values
        """
        return np.mean(np.power(y - predictions, 2))

    def model_backward(self, X, y, predictions):
        """
        Computes the gradients of weights and bias for the model.

        Input:
        - X: Input data
        - y: Actual target values
        - predictions: Model predictions

        Output:
        - gradient_bias: Gradient of bias
        - gradient_weights: Gradient of weights
        """
        gradient_weights = (-2 / len(y)) * np.dot(X.T,y -  predictions)
        gradient_bias = (-2 / len(y)) * np.sum(y- predictions)
        return gradient_bias, gradient_weights
    
    def update_parameters(self, gradient_bias, gradient_weights, learning_rate):
        """
        Updates the parameters of the model based on gradients and learning rate.

        Input:
        - gradient_bias: Gradient of bias
        - gradient_weights: Gradient of weights
        - learning_rate: Learning rate

        Output:
        - None
        """
        self.b -= learning_rate * gradient_bias
        self.W -= learning_rate * gradient_weights
        
    def predict(self, X):
        """
        Performs predictions using the trained model. 
        (Wrapper-function for model_forward(X))

        Input:
        - X: Input data for prediction, shaped as (number of data points, number of features)

        Output:
        - predictions: Model predictions for the given input data
        """
        return self.model_forward(X)
    
    def train_linear_model(self, X, y, learning_rate, num_iterations):
        """
        Trains the linear regression model using batch gradient descent to minimize the loss.
    
        Inputs:
        - X: Input data
        - y: Actual target values
        - learning_rate: Learning rate for gradient descent
        - num_iterations: Number of iterations for training
    
        Outputs:
        - cost_history: List of costs during training.
        """
        
        num_features = X.shape[1]
        self.initialize_parameters(num_features)

        cost_history = []
        
        for iter  in range(num_iterations):
            
            predictions = self.predict(X)
            
            mse_cost = self.compute_cost(y, predictions)
            cost_history.append(mse_cost)
            
            gradient_bias, gradient_weights = self.model_backward(X, y, predictions)
            
            self.update_parameters(gradient_bias, gradient_weights, learning_rate)
            
            # Printing cost every 100 iterations
            if iter % 100 == 0:
                print(f"Iteration {iter}, Cost: {mse_cost}")
            
        return cost_history

    