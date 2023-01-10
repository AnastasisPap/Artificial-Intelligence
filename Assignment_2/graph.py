import matplotlib.pyplot as plt
import numpy as np

def accuracy_graph(accuracy, title):
    x = list(range(10, 101, 10))
    y = accuracy
    plt.xlabel('Training set %')
    plt.ylabel('Error %')
    plt.title(title)
    plt.plot(x, y)
    plt.show()