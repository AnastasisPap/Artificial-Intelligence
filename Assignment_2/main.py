import tensorflow as tf
import numpy as np
from naive_bayes import *
import log_reg
from sklearn.model_selection import train_test_split 
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
    percentage_increase = 20 
    val_size = 0.2

    print('=' * 15 + ' Naive Bayes ' + '=' * 15)
    x_train_split, x_val, y_train_split, y_val = train_test_split(x_train_binary, y_train, test_size=val_size, random_state=55)
    #evaluate_bayes((x_train_split, y_train_split), (x_val, y_val), percentage_increase)
    # evaluate_bayes((x_train_binary, y_train), (x_test_binary, y_test), percentage_increase)
    print('=' * 15 + ' Logistic Regression ' + '=' * 15)
    # Resplit them to avoid bias (they are shuffled)
    x_train_split, x_val, y_train_split, y_val = train_test_split(x_train_binary, y_train, test_size=val_size, random_state=111)
    #log_reg.evaluate_logistic_regression((x_train_split, y_train_split), (x_val, y_val), percentage_increase, 100)
    print('=' * 15 + ' RNN (with LSTM) ' + '=' * 15)
    x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=253)
    rnn((x_train_split, y_train_split), (x_val, y_val), m, percentage_increase, 2)

main(20000, 1000, 20000)