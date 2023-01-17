import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
import fasttext
import numpy as np

def main(n, m, k):
    index_from = n
    seed = 113
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
    word_index = tf.keras.datasets.imdb.get_word_index()
    index2word = dict((i+3, word) for (word, i) in word_index.items())
    index2word[0] = '[pad]'
    index2word[1] = '[bos]'
    index2word[2] = '[oov]'
    x_train = np.array([' '.join(index2word[i] if i in index2word else '' for i in review) for review in x_train])
    x_test = np.array([' '.join(index2word[i] if i in index2word else '' for i in review) for review in x_test])
    avg_length = get_average_length(x_train)
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=m, output_mode='int', ngrams=1, name='vector_text', output_sequence_length=avg_length)
    vectorizer.adapt(x_train)

    fasttext_model = fasttext.load_model('cc.en.300.bin')
    embeddings_matrix = np.zeros(shape=(len(vectorizer.get_vocabulary()), 300))

    for i, word in enumerate(vectorizer.get_vocabulary()):
        embeddings_matrix[i] = fasttext_model.get_word_vector(word=word)
    del fasttext_model

    model = rnn(vectorizer, embeddings_matrix, emb_size=300)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['binary_accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10, verbose=1, batch_size=32)
    print(model.evaluate(x_test, y_test))

def get_average_length(x_train):
    avg_len = 0
    for example in x_train: avg_len += len(str(example).split())
    return int(avg_len / len(x_train))

def rnn(vectorizer, embeddings_matrix, num_layers=1, emb_size=64, h_size=64):
    inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='txt_input')
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(
                                    input_dim=len(vectorizer.get_vocabulary()),
                                    output_dim=emb_size, name='word_embeddings',
                                    trainable=False, weights=[embeddings_matrix],
                                    mask_zero=True)(x)
    
    x = tf.keras.layers.Dropout(rate=0.25)(x)
    for n in range(num_layers):
        if n != num_layers - 1:
            x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=h_size,
                                                name=f'bigru_cell_{n}',
                                                return_sequences=True,
                                                dropout=0.2))(x)
        else:
            x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=h_size,
                                                name=f'bigru_cell_{n}',
                                                dropout=0.2))(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    o = tf.keras.layers.Dense(units=1, activation='sigmoid', name='lr')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=o, name='simple_rnn')

main(20000, 5000, 20000)
