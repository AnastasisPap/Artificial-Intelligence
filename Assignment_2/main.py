import tensorflow as tf
import numpy as np
from naive_bayes import *
from logistic_regression import *

# TODO: Fix vocab creation
def preprocessing(n, m, k):
    # Use the default parameters to keras.datasets.imdb.load_data
    start_char = 1
    oov_char = 2
    index_from = n
    seed = 113
    # Retrieve the training sequences.
    (x, y), (x_test, y_test)= tf.keras.datasets.imdb.load_data(
        start_char=start_char, oov_char=oov_char, index_from=index_from, seed=seed
    )
    
    # Split inital training data into training data and dev data, proportions: (1-split)/split
    (x_train, y_train), (x_dev, y_dev) = split_training_data(x, y, 0.2)

    # Retrieve the word index file mapping words to indices
    word_index = tf.keras.datasets.imdb.get_word_index()
    # Reverse the word index to obtain a dict mapping indices to words
    # And add `index_from` to indices to sync with `x_train`
    inverted_word_index = dict(
        (i + index_from, word) for (word, i) in word_index.items()
    )

    #inverted_word_index[start_char] = "[START]"
    #inverted_word_index[oov_char] = "[OOV]"

    #decoded_sequence = " ".join(inverted_word_index[i] for i in x_test[4])
    idxes = list(inverted_word_index.keys())
    idxes.sort()
    idxes = idxes[:-k][:m]
    
    x_train_binary = transform_vector(x_train, idxes)
    x_test_binary = transform_vector(x_test, idxes)
    # evaluate_bayes((x_train_binary, y_train), (x_test_binary, y_test))
    evaluate_logistic_regression((x_train_binary, y_train), (x_test_binary, y_test))

def transform_vector(x, idxes):
    matrix = []
    for sample in x:
        vector = np.zeros(len(idxes)) 
        for idx in sample:
            if idxes[0] <= idx <= idxes[-1]:
                vector[idx - idxes[0]] = 1
    
        matrix.append(vector)

    return np.array(matrix)

# Split inital training data into training data and dev data, proportions: (1-split)/split
def split_training_data(x, y, split):
    pos, neg = (split * len(x))//2, (split * len(x))//2
    x_train, y_train = [], []
    x_dev, y_dev = [], []
    
    for i in range(len(x)):
        if (pos > 0 and y[i] == 1):
            x_dev.append(x[i])
            y_dev.append(y[i])
            pos -= 1
        elif (neg > 0 and y[i] == 0):
            x_dev.append(x[i])
            y_dev.append(y[i])
            neg -= 1
        else:
            x_train.append(x[i])
            y_train.append(y[i])

    return (x_train, y_train), (x_dev, y_dev)
    

preprocessing(20000, 1000, 20000)
