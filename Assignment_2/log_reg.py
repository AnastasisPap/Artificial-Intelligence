import numpy as np
from random import sample
import tensorflow as tf
from graph import *
from logistic_regression import *
from tqdm import tqdm

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(x_train, y_train, batch_size, epochs, learning_rate, lambda_param):
    m, n = x_train.shape

    w = np.zeros((n, 1))
    y_train = y_train.reshape(m, 1)

    for epoch in tqdm(range(epochs)):
        for i in range((m - 1) // batch_size + 1):
            start = i * batch_size
            x_train_batch = x_train[start:start + batch_size]
            y_train_batch = y_train[start:start + batch_size]

            p = sigmoid(np.dot(x_train_batch, w))
            w += learning_rate * np.dot(x_train_batch.T, (y_train_batch - p)) - 2 * learning_rate * lambda_param * w

    return w

def predict(x, w):
    preds = np.dot(x, w)
    pred_class = [1 if i >= 0 else 0 for i in preds]

    return np.array(pred_class)

def evaluate_logistic_regression(training_set, test_set):
    x_train, y_train = training_set
    x_test, y_test = test_set
    x_train = np.insert(x_train, 0, 1, axis=1)
    x_test = np.insert(x_test, 0, 1, axis=1)
    
    training_error = []
    test_error = []

    for i in range(1, 11):
        set_sample = list(sample(list(range(len(x_train))), int(len(x_train) * 0.1 * i)))
        x_train_sample = np.array([x_train[i] for i in set_sample])
        y_train_sample = np.array([y_train[i] for i in set_sample])
        print(f'Training on {i * 10}%')
        w = train(x_train_sample, y_train_sample, batch_size=100, epochs=50, learning_rate=0.01, lambda_param=0)

        training_error.append(np.sum(y_train_sample == predict(x_train_sample, w))/len(y_train_sample))
        test_error.append(np.sum(y_test == predict(x_test, w)) / len(y_train))
    
    accuracy_graph(training_error, 'Training set curve')
    accuracy_graph(test_set, 'Test set curve')