import pandas as pd
import numpy as np
from random import sample
from graph import *
from tqdm import tqdm

class Training:
    def __init__(self, pos_probs, neg_probs, p_1):
        self.pos_probs = pos_probs
        self.neg_probs = neg_probs
        self.p_0 = 1 - p_1
        self.p_1 = p_1

def train_naive_bayes(dataset):
    x_train, y_train = dataset
    examples, _ = x_train.shape
    pos_counts = np.count_nonzero(y_train == 1)
    p_1 = pos_counts / examples
    pos_probs = (np.sum(x_train[np.where(y_train == 1)], axis=0) + 1) / (pos_counts + 2)
    neg_probs = (np.sum(x_train[np.where(y_train == 0)], axis=0) + 1) / (examples - pos_counts + 2)

    return Training(pos_probs, neg_probs, p_1)

def test_naive_bayes(x, training):
    p_0 = training.p_0 * np.prod(np.where(x == 1, training.neg_probs, 1 - training.neg_probs), axis=1)
    p_1 = training.p_1 * np.prod(np.where(x == 1, training.pos_probs, 1 - training.pos_probs), axis=1)
    
    return p_1 > p_0

def calculate_metrics(x, y, w):
    error = 1 - (np.sum(y == test_naive_bayes(x, w)) / len(y))
    TP = np.sum(np.logical_and(y == 1, test_naive_bayes(x, w) == 1))
    FP = np.sum(np.logical_and(y == 0, test_naive_bayes(x, w) == 1))
    FN = np.sum(np.logical_and(y == 1, test_naive_bayes(x, w) == 0))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return error, precision, recall

def evaluate_bayes(training_set, test_set, perc):
    x_train, y_train = training_set
    x_test, y_test = test_set
    training_error = []
    training_precision = []
    training_recall = []

    test_error = []
    test_precision = []
    test_recall = []
    for i in tqdm(range(perc, 101, perc)):
        set_sample = sample(list(range(len(x_train))), int(len(x_train) * i * 0.01))
        x_train_sample = np.array([x_train[i] for i in set_sample])
        y_train_sample = np.array([y_train[i] for i in set_sample])
        training = train_naive_bayes((x_train_sample, y_train_sample))

        training_metrics = calculate_metrics(x_train_sample, y_train_sample, training)
        training_error.append(training_metrics[0])
        training_precision.append(training_metrics[1])
        training_recall.append(training_metrics[2])

        test_metrics = calculate_metrics(x_test, y_test, training)
        test_error.append(test_metrics[0])
        test_precision.append(test_metrics[1])
        test_recall.append(test_metrics[2])
    
    F_1 = (2 * np.array(test_precision) * np.array(test_recall)) / (np.array(test_precision) + np.array(test_recall))
    table = np.array([[1-v for v in test_error], test_precision, test_recall, F_1])
    cols = [str(a)+'%' for a in range(perc, 101, perc)]
    idx = ['Accuracy', 'Precision', 'Recall', 'F1']
    df = pd.DataFrame(table, columns=cols, index=idx)

    print('\n\n=== Metrics for test set ===')
    print(df)
    accuracy_graph(training_error, test_error, perc)
    prec_rec_graph(training_precision, training_recall, perc, 'Training')
    prec_rec_graph(test_precision, test_recall, perc, 'Test', list(F_1))