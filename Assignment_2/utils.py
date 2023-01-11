import numpy as np
import pandas as pd
from graph import *

def calculate_metrics(x, y, training, predict_func, clf=None):
    prediction = predict_func(x, training) if clf is None else clf.predict(x)
    error = 1 - (np.sum(y == prediction) / len(y))
    TP = np.sum(np.logical_and(y == 1, prediction == 1))
    FP = np.sum(np.logical_and(y == 0, prediction == 1))
    FN = np.sum(np.logical_and(y == 1, prediction == 0))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return error, precision, recall

def display_metrics(metrics, perc, title):
    F_1 = (2 * metrics[:, 1] * metrics[:, 2]) / (metrics[:, 1] + metrics[:, 2])
    table = np.array([[1-v for v in metrics[:, 0]], metrics[:, 1], metrics[:, 2], F_1])
    cols = [str(a)+'%' for a in range(perc, 101, perc)]
    idx = ['Accuracy', 'Precision', 'Recall', 'F1']
    df = pd.DataFrame(table, columns=cols, index=idx)

    print(f'\n\n=== Metrics for {title} ===')
    print(df)
    prec_rec_graph(metrics[:, 1], metrics[:, 2], perc, title)