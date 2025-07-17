# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:24:48 2024

Softmax Regression Model Implementation using batch gradient descent -
used given load_data function to read and preprocess the dataset:
    - handling NaN values
    - extracting the features as input and 'origin' as output for classification
    
The model consists of methods for initializing parameters, performing forward computation,
computing the cost (Cross Entropy), computing gradients, updating parameters, making predictions,
and training the model.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SoftmaxRegression():
    def initialize_parameters(self, num_features, num_classes):
        """
        Initializes the parameters (weights and bias) of the softmax regression model with zeros.

        Input:
        - num_features: Number of features in the input data
        - num_classes: Number of classes in the classification problem

        Output:
        - None
        """
        self.classes = num_classes
        self.b = np.zeros(num_classes) #size (3,)
        self.W = np.zeros((num_features, num_classes)) #size(7,3)
    
    def model_forward(self, X):
        """
        Performs forward computation for the softmax regression model.
    
        Input:
        - X: Input data (number of data points, number of features)
    
        Output:
        - predictions: Model predictions for the given input data
        """
        return (np.dot(X, self.W) + self.b)

    def model_backward(self, X, y, predictions):
        """
        Computes the gradients of weights and bias for the softmax regression model using backward propagation.
    
        Input:
        - X: Input data
        - y: Actual target values
        - predictions: Model predictions
    
        Output:
        - gradient_bias : , dimension change compared to Linear Model 
        - gradient_weights
        """
        gradient_weights = (-2 / len(y)) * np.dot(X.T,y -  predictions)
        gradient_bias = (-2 / len(y)) * np.sum(y- predictions, axis=0) #change dim
        return gradient_bias, gradient_weights
    
    def update_parameters(self, gradient_bias, gradient_weights, learning_rate):
        """
        Updates the parameters (weights and bias) of the softmax regression model using gradient descent.
    
        Input:
        - gradient_bias
        - gradient_weights
        - learning_rate
    
        Output:
        - None
        """
        self.b -= learning_rate * gradient_bias
        self.W -= learning_rate * gradient_weights
        
    def predict(self, X):
        """
        Predicts the class probabilities for the input data using the softmax function.

        Input:
            - X: Input data

        Output:
            - predictions: Predicted class probabilities for the given input data
            - calls softmax activation function for predictions
        """
        return self.softmax(self.model_forward(X))
    
    def softmax(self, z):
        """
        Computes the softmax activation function for the given input.
    
        Input:
        - z: Input values, shaped as (number of data points, number of classes)
    
        Output:
        - softmax_output: Resulting softmax activations for the input data
        """
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T
    
    def to_classlabel(self, z):
        """
        Converts the softmax output into class labels.
    
        Input:
        - z: Softmax output probabilities, shaped as (number of data points, number of classes)
    
        Output:
        - class_labels: Predicted class labels based on the highest probability
        """
        return z.argmax(axis=1)
    
    def cross_entropy(self, sm_output, y_target):
        """
        Computes the cross-entropy loss between the softmax output and the target labels.
    
        Input:
        - sm_output: Softmax output probabilities, shaped as (number of data points, number of classes)
        - y_target: Target labels, one-hot encoded, shaped as (number of data points, number of classes)
    
        Output:
        - cross_entropy_loss: Cross-entropy loss between the softmax output and the target labels 
            shaped as (number of data points, number of classes)
        """
        return - np.sum(np.log(sm_output) * (y_target), axis=1)
    
    def compute_cost(self, output, y_target):
        """
        Computes the mean cross-entropy loss as the cost function.
    
        Input:
        - cross_entropy_loss 
        Output:
        - cost: Mean cross-entropy loss
        """
        return np.mean(self.cross_entropy(output, y_target))
    
    def train_softmax_model(self, X, y, learning_rate, num_iterations):
        """
        Trains the softmax regression model using gradient descent.
    
        Input:
        - X: Input data
        - y: Target labels, one-hot encoded
        - learning_rate: Learning rate for gradient descent
        - num_iterations: Number of iterations for training
    
        Output:
        - cost_history: List containing the cost (cross-entropy loss) at each iteration
        - y_hat_classes: Predicted class labels for the input data after training
        """
            
        num_features = X.shape[1]
        self.initialize_parameters(num_features, 3)

        cost_history = []
        
        for iter  in range(num_iterations):
            
            predictions = self.predict(X)
            # argmax
            y_hat_classes = self.to_classlabel(predictions)
            
            # use softmax predictions for cost calculation
            cost = self.compute_cost(predictions, y)
            cost_history.append(cost)
            
            gradient_bias, gradient_weights = self.model_backward(X, y, predictions)
            
            self.update_parameters(gradient_bias, gradient_weights, learning_rate)
            
            # Printing cost every 100 iterations
            if iter % 100 == 0:
                cost_rounded = round(cost, 7)
                train_accuracy = accuracy_score(y.argmax(axis=1), y_hat_classes)
                train_accuracy_rounded = round(train_accuracy, 7)
                print(f"Iteration {iter},\tCE_Cost: {cost_rounded},\tTrain Accuracy: {train_accuracy_rounded}")
        return cost_history, y_hat_classes
    

def one_hot(y, c): 
    """
    Converts target labels into one-hot encoded format.

    Input:
    - y: Target labels (ground truth)
    - c: Number of classes

    Output:
    - y_hot: One-hot encoded representation of the target labels
    """
    y_hot = np.zeros((len(y), c))
    y_hot[np.arange(len(y)), y] = 1
    return y_hot
    