import matplotlib.pyplot as plt
import argparse
import sys

import numpy as np
import pandas as pd

from softmax_gradient_descent import one_hot, train_test_split, SoftmaxRegression, accuracy_score
from linear_gradient_descent import LinearRegression

def load_auto(mode = "linear_univariate"):
	Auto = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()
    
	# Extract relevant data features
	if mode == "linear_univariate":
		X_train = Auto[['horsepower']].values
		Y_train = Auto[['mpg']].values
	
	# Multivariate Regression
	elif mode == "linear_multivariate":
		X_train = Auto[['cylinders','displacement','horsepower','weight', 'acceleration','year','origin']].values
		Y_train = Auto[['mpg']].values

	# Multiclass - softmax
	else:
		X_train = Auto[['cylinders','displacement','horsepower','weight', 'acceleration','year','mpg']].values
		Y_train = Auto[['origin']].values

	return X_train, Y_train


def main():
    parser = argparse.ArgumentParser(description="Train a model based on the selected mode.")
    parser.add_argument(
        "--mode", 
        choices=["linear_univariate", "linear_multivariate", "softmax"],
        required=True,
        help="Select the training mode."
    )
    args = parser.parse_args()
    mode = args.mode

    print(f"Running in mode: {mode}")
    X_train, Y_train = load_auto(mode)
    
    if mode == "softmax":
		## Pre-Processing
        Y_train = Y_train.flatten()
        # change 1,2,3 to class 0,1,2
        Y_train -=1

        # One-Hot Encoding
        Y_train_coded = one_hot(Y_train,3)
        
        # X - Normalizing
        X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        
        # Split Test/Train Data
        X_train, X_test, Y_train_coded, Y_test_coded = train_test_split(X_train, Y_train_coded, test_size=0.3, random_state=42)
        
        # Parameter
        learning_rate = 1e-2
        num_iterations = 1000
        
        regression = SoftmaxRegression()
        cost_history, y_hat_classes = regression.train_softmax_model(X_train, Y_train_coded, learning_rate, num_iterations)

        print("Optimized Weights:", regression.W)
        print("Optimized Bias:", regression.b)
        print("Cross_Entropy_Loss:", cost_history[-1])
        
        # Test Data
        test_predictions = regression.predict(X_test).argmax(axis=1)
        # ground_truth to 1 dim array
        y_test_uncoded = np.argmax(Y_test_coded, axis=1)
        test_accuracy = accuracy_score(y_test_uncoded, test_predictions)
        
        test_size = len(X_test)
        train_size = len(X_train)
        
        print("Train Size:", train_size)
        print("Test Size:", test_size)
        print("Accuracy_Test:", test_accuracy)
		
    else:
		## Pre-Processing
        Y_train = Y_train.flatten()
        # Normalizing input data
        mean = np.mean(X_train, axis=0) 
        std = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std

        # Test - ONE SAMPLE 
        # X_train = X_train[0].reshape(1, -1)
        # Y_train = Y_train[0].reshape(1, )

        # Parameter
        learning_rate = 1e-2
        num_iterations = 1000

        regression = LinearRegression()
        cost_history = regression.train_linear_model(X_train, Y_train, learning_rate, num_iterations)

        print("Optimized Weights:", regression.W)
        print("Optimized Bias:", regression.b)
        print("MSE_Loss:", cost_history[-1])
		

if __name__ == "__main__":
    main()
	
