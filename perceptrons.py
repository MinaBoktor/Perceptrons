from random import random
import numpy as np


def SLP(train_df, test_df, use_bias=True, learning_rate=0.1, epochs=1000):

    # Initialize random weights for the features
    weights = np.random.rand(train_df.shape[1] - 1)/5000

    x_train = train_df.drop(columns=["Species"]).to_numpy()
    y_train = train_df["Species"].to_numpy()

    x_test = test_df.drop(columns=["Species"]).to_numpy()
    y_test = test_df["Species"].to_numpy()

    if use_bias:
        bias = random()/5000
    else:
        bias = 0

    errors = [] # NEW: Initialize a list to track errors

    for _ in range(epochs):
        misclassifications = 0

        for row in range(train_df.shape[0]):
            y_predict = signum(np.dot(weights.T, x_train[row])+bias)
            if (y_predict != y_train[row]):
                loss = y_train[row] - y_predict
                weights += learning_rate * loss * x_train[row]
                if use_bias:
                    bias += learning_rate * loss
                misclassifications += 1
        errors.append(misclassifications)

    y_pred = []
    for row in range(test_df.shape[0]):
        y_predict = signum(np.dot(weights.T, x_test[row])+bias)
        y_pred.append(y_predict)
    return weights, bias, accuracy(test_df, weights, bias), y_test, y_pred, errors


def adaline(train_df, test_df, learning_rate=0.1, epochs=1000, mse_threshold=0.15):

    # Initialize random weights for the features
    weights = np.random.rand(train_df.shape[1] - 1)/5000

    x_train = train_df.drop(columns=["Species"]).to_numpy()
    y_train = train_df["Species"].to_numpy()

    x_test = test_df.drop(columns=["Species"]).to_numpy()
    y_test = test_df["Species"].to_numpy()

    errors = []

    for _ in range(epochs):

        for row in range(train_df.shape[0]):
            y_predict = np.dot(weights.T, x_train[row])
            loss = y_train[row] - y_predict
            weights += learning_rate * loss * x_train[row]

        # Calculate MSE
        square_error = 0
        for row in range(train_df.shape[0]):
            loss = y_train[row] - np.dot(weights.T, x_train[row])
            square_error += loss**2

        mse = 0.5 * square_error / train_df.shape[0]

        errors.append(mse)

        if mse < mse_threshold:
            break


    y_pred = []
    for row in range(test_df.shape[0]):
        y_predict = signum(np.dot(weights.T, x_test[row]))
        y_pred.append(y_predict)
    return weights, 0, accuracy(test_df, weights), y_test, y_pred, errors


def accuracy(test_df, weights, bias=0):
    x_test = test_df.drop(columns=["Species"]).to_numpy()
    y_test = test_df["Species"].to_numpy()

    correct_predictions = 0
    for row in range(test_df.shape[0]):
        y_predict = signum(np.dot(weights.T, x_test[row])+bias)
        if (y_predict == y_test[row]):
            correct_predictions += 1
    return correct_predictions / test_df.shape[0]


def signum(x):
    return np.sign(x)
