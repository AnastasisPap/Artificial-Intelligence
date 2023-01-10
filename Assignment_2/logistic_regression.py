from random import sample
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from graph import *

learning_rate = 0.01
def train_logistic_regression(training_set, max_iter, lambda_param):
    x_train, y_train = training_set
    n = len(x_train[0])
    W = np.zeros(n)
    
    for i in tqdm(range(max_iter)):
        j = 0
        while j < len(x_train):
            W = update_weights(W, x_train[j:j+100] , y_train[j:j+100], lambda_param)
            j += 100
        
    return W 

def test_logistic_regression(x, W):
    product = np.dot(x, W)

    return 0 if product < 0 else 1

def evaluate_logistic_regression(training_set, test_set):
    x_test, y_test = test_set
    x_train, y_train = training_set
    x_train = np.insert(x_train, 0, 1, axis=1)
    x_test = np.insert(x_test, 0, 1, axis=1)
    training_error = []
    test_error = []
    
    count_test = 0
    for i in range(1, 11):
        #set_sample = list(sample(list(range(len(x_train))), int(len(x_train) * i * 0.1)))
        x_train_sample = x_train[:int(len(x_train) * i * 0.1)]
        y_train_sample = y_train[:int(len(x_train) * i * 0.1)]
        print(f'Training on {i * 10}%')
        W = train_logistic_regression((x_train_sample, y_train_sample), 200, 0.4)
        count_training = 0
        for i in range(len(x_train_sample)):
            res = test_logistic_regression(x_train_sample[i], W)
            count_training += 1 if res == y_train_sample[i] else 0
        training_error.append(count_training/len(x_train_sample))

        count_test = 0
        for i in range(len(x_test)):
            res = test_logistic_regression(x_test[i], W)
            count_test += 1 if res == y_test[i] else 0
        test_error.append(count_test/len(x_test))
        print(test_error)

    accuracy_graph(training_error, 'Training set curve')
    accuracy_graph(test_error, 'Test set curve')

def update_weights(W, x_i, y_i, lambda_param):
    p_i_1 = sigmoid(np.dot(x_i, W))
    W += learning_rate * np.matmul((y_i - p_i_1), x_i) - 2 * learning_rate * lambda_param * W
    
    return W

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regr(x_train, y_train, x_test, y_test):
    x_train = [[v for v in row] for row in x_train]
    y_train = [row for row in y_train]
    x_test = [[v for v in row] for row in x_test]
    y_test = [row for row in y_test]

    log_keras = tf.keras.models.Sequential([tf.keras.layers.Dense(units=1, activation='sigmoid')])
    log_keras.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(), metrics=['binary_accuracy'])
    log_keras.fit(x=x_train, y=y_train, epochs=200, verbose=1, batch_size=1)
    print(log_keras.evaluate(x_test, y_test))