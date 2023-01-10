import tensorflow as tf
import numpy as np
from random import sample
from graph import *
from tqdm import tqdm

class Training:
    def __init__(self, probabilities, count_0):
        self.probabilities = probabilities
        self.count_0 = count_0
        self.count_1 = 1 - count_0

def train_naive_bayes(dataset):
    x, y = dataset
    counts = np.zeros((2, len(x[0])))

    for i in tqdm(range(len(x))):
        counts[y[i]] += x[i]
    
    count_0 = y.count(0)
    probabilities = counts + 1
    probabilities[0] /= (count_0 + 2)
    probabilities[1] /= (len(y) - count_0 + 2)
    count_0 /= len(y)

    return Training(probabilities, count_0)

def test_naive_bayes(x, training):
    p_0 = training.count_0
    for i, bit in enumerate(x):
        p_0 *= training.probabilities[0][i] if bit == 1 else 1 - training.probabilities[0][i]
    
    p_1 = training.count_1
    for i, bit in enumerate(x):
        p_1 *= training.probabilities[1][i] if bit == 1 else 1 - training.probabilities[1][i]
    
    return 1 if p_1 > p_0 else 0

def evaluate_bayes(training_set, test_set):
    x_train, y_train = training_set
    x_test, y_test = test_set
    training_error = []
    test_error = []
    for i in range(1, 11):
        set_sample = sample(list(range(len(x_train))), int(len(x_train) * i * 0.1))
        x_train_sample = [x_train[i] for i in set_sample]
        y_train_sample = [y_train[i] for i in set_sample]
        print(f'Training on {i * 10}%')
        training = train_naive_bayes((x_train_sample, y_train_sample))
        count_training = 0
        for i in range(len(set_sample)):
            res = test_naive_bayes(x_train_sample[i], training)
            count_training += 1 if res == y_train_sample[i] else 0
        training_error.append(count_training/len(set_sample))

        count_test = 0
        for i in range(len(x_test)):
            res = test_naive_bayes(x_test[i], training)
            count_test += 1 if res == y_test[i] else 0
        test_error.append(count_test / len(x_test))
        print(test_error)
    
    accuracy_graph(training_error, 'Training set curve')
    accuracy_graph(test_error, 'Test set curve')
