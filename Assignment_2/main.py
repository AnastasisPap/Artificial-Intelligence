import tensorflow as tf
import numpy as np
from naive_bayes import *
import log_reg
from sklearn.feature_extraction.text import CountVectorizer
from MLP import rnn

def main(n, m, k):
    index_from = n
    seed = 113
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(index_from=index_from, seed=seed)
    word_index = tf.keras.datasets.imdb.get_word_index()
    index2word = dict((i+3, word) for (word, i) in word_index.items())
    index2word[0] = '[pad]'
    index2word[1] = '[bos]'
    index2word[2] = '[oov]'
    max_idx = max(index2word)
    idxes = list(range(max_idx, max_idx-k, -1))
    for idx in idxes: del index2word[idx]

    x_train = np.array([' '.join([index2word[idx] if idx in index2word else '[oov]' for idx in text]) for text in x_train])
    x_test = np.array([' '.join([index2word[idx] if idx in index2word else '[oov]' for idx in text]) for text in x_test])
    binary_vectorizer = CountVectorizer(binary=True, max_features=m)
    x_train_binary = binary_vectorizer.fit_transform(x_train).toarray()
    x_test_binary = binary_vectorizer.transform(x_test).toarray()

    # print('=' * 15 + ' Naive Bayes ' + '=' * 15)
    # evaluate_bayes((x_train_binary, y_train), (x_test_binary, y_test), 10)
    # print('=' * 15 + ' Logistic Regression ' + '=' * 15)
    # log_reg.evaluate_logistic_regression((x_train_binary, y_train), (x_test_binary, y_test), 10, 100)
    # print('=' * 15 + ' RNN (with LSTM) ' + '=' * 15)
    rnn((x_train, y_train), (x_test, y_test), m, 100)

main(20000, 1000, 20000)