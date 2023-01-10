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

        training_error.append(1 - (np.sum(y_train_sample == predict(x_train_sample, w))/len(y_train_sample)))
        test_error.append(1 - (np.sum(y_test == predict(x_test, w)) / len(y_test)))
        TP_train = np.sum(np.logical_and(y_train_sample == 1, predict(x_train_sample, w) == 1))
        FP_train = np.sum(np.logical_and(y_train_sample == 0, predict(x_train_sample, w) == 1))
        FN_train = np.sum(np.logical_and(y_train_sample == 1, predict(x_train_sample, w) == 0))
        TP_test = np.sum(np.logical_and(y_test == 1, predict(x_test, w) == 1))
        FP_test = np.sum(np.logical_and(y_test == 0, predict(x_test, w) == 1))
        FN_test = np.sum(np.logical_and(y_test == 1, predict(x_test, w) == 0))

        training_precision.append(TP_train / (TP_train + FP_train))
        training_recall.append(TP_train / (TP_train + FN_train))
        test_precision.append(TP_test / (TP_test + FP_test))
        test_recall.append(TP_test / (TP_test + FN_test))
    F_1 = (2 * test_precision * test_recall) / (test_precision + test_recall)

    accuracy_graph(training_error, test_error, percentage_increase)
    prec_rec_graph(training_precision, training_recall, percentage_increase, 'Training')
    prec_rec_graph(test_precision, test_recall, percentage_increase, 'Test', F_1)