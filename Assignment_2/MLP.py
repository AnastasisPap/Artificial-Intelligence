import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tqdm import tqdm
from random import sample
from sklearn.metrics import classification_report
import numpy as np

def rnn(train_set, test_set, max_vocab, perc):
    x_train, y_train = train_set
    x_test, y_test = test_set
    training_metrics = []
    test_metrics = []

    for i in tqdm(range(perc, 101, perc)):
        set_sample = list(sample(list(range(len(x_train))), int(len(x_train) * 0.01 * i)))
        x_sample = np.array([x_train[i] for i in set_sample])
        y_sample = np.array([y_train[i] for i in set_sample])

        model = create_model((x_sample, y_sample), max_vocab)
        print(model.evaluate(x_test, y_test)[-1])

def get_average_length(x_train):
    avg_len = 0
    for example in x_train: avg_len += len(str(example).split())
    return int(avg_len / len(x_train))

def create_model(train, max_vocab):
    x_train, y_train = train
    avg_length = get_average_length(x_train)
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_vocab, output_sequence_length=avg_length)
    vectorizer.adapt(x_train)

    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(
            input_dim=len(vectorizer.get_vocabulary()),
            output_dim=64,
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])
    
    model.fit(x=x_train, y=y_train, epochs=1, verbose=1, batch_size=100)

    return model
