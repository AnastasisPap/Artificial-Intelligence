import numpy as np
from random import sample
import tensorflow as tf
from graph import *
from logistic_regression import *
import pandas as pd
from tqdm import tqdm

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(x_train, y_train, batch_size, epochs, learning_rate, lambda_param):
    m, n = x_train.shape

    w = np.zeros((n, 1))
    y_train = y_train.reshape(m, 1)

    for epoch in range(epochs):
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

def calculate_metrics(x, y, w):
    error = 1 - (np.sum(y == predict(x, w)) / len(y))
    TP = np.sum(np.logical_and(y == 1, predict(x, w) == 1))
    FP = np.sum(np.logical_and(y == 0, predict(x, w) == 1))
    FN = np.sum(np.logical_and(y == 1, predict(x, w) == 0))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return error, precision, recall

def evaluate_logistic_regression(training_set, test_set, percentage_increase):
    x_train, y_train = training_set
    x_test, y_test = test_set
    x_train = np.insert(x_train, 0, 1, axis=1)
    x_test = np.insert(x_test, 0, 1, axis=1)
    
    training_error = []
    test_error = []
    training_precision = []
    test_precision = []
    training_recall = []
    test_recall = []

    for i in tqdm(range(percentage_increase, 101, percentage_increase)):
        set_sample = list(sample(list(range(len(x_train))), int(len(x_train) * 0.01 * i)))
        x_train_sample = np.array([x_train[i] for i in set_sample])
        y_train_sample = np.array([y_train[i] for i in set_sample])
        w = train(x_train_sample, y_train_sample, batch_size=100, epochs=100, learning_rate=0.01, lambda_param=0.001)

        training_metrics = calculate_metrics(x_train_sample, y_train_sample, w)
        training_error.append(training_metrics[0])
        training_precision.append(training_metrics[1])
        training_recall.append(training_metrics[2])

        test_metrics = calculate_metrics(x_test, y_test, w)
        test_error.append(test_metrics[0])
        test_precision.append(test_metrics[1])
        test_recall.append(test_metrics[2])

    F_1 = (2 * np.array(test_precision) * np.array(test_recall)) / (np.array(test_precision) + np.array(test_recall))
    table = np.array([[1-v for v in test_error], test_precision, test_recall, F_1])
    cols = [str(a)+'%' for a in range(percentage_increase, 101, percentage_increase)]
    idx = ['Accuracy', 'Precision', 'Recall', 'F1']
    df = pd.DataFrame(table, columns=cols, index=idx)

    print('\n\n=== Metrics for test set ===')
    print(df)
    accuracy_graph(training_error, test_error, percentage_increase)
    prec_rec_graph(training_precision, training_recall, percentage_increase, 'Training')
    prec_rec_graph(test_precision, test_recall, percentage_increase, 'Test', list(F_1))