import matplotlib.pyplot as plt
import numpy as np

def accuracy_graph(train_error, test_error, perc):
    x = list(range(perc, 101, perc))
    y = train_error
    z = test_error
    plt.xlabel('Dataset size %')
    plt.ylabel('Error %')
    plt.title('Error VS. dataset size')
    plt.plot(x, y, color='g', label='Train curve')
    plt.plot(x, z, color='r', label='Test curve')
    plt.legend()
    plt.show()

def prec_rec_graph(precision, recall, perc, type, F_measure = None):
    x = list(range(perc, 101, perc))
    y = precision
    z = recall
    plt.xlabel('Dataset size %')
    plt.title(f'Precision/Recall VS. dataset size ({type})')
    plt.plot(x, y, 'g-', label='Precision')
    plt.plot(x, z, 'r--', label='Recall')
    if F_measure: plt.plot(x, F_measure, 'b-', label='F1')
    plt.legend()
    plt.show()