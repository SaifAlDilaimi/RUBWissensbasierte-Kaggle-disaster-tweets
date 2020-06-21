import tweet_tokenizer
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import string
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Embedding, Dropout

# To run this you need to download the pretrained embeddings: http://nlp.stanford.edu/data/glove.twitter.27B.zip

def load_dataset():
    df_train = pd.read_csv('train.csv', dtype={'id': np.int16, 'target': np.int8})
    df_test = pd.read_csv('test.csv', dtype={'id': np.int16})

    print('Training Set Shape = {}'.format(df_train.shape))
    print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
    print('Test Set Shape = {}'.format(df_test.shape))
    print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))

    return df_train, df_test


def get_vectorizer(df_train, df_test):
    
    # Vectorizes and pads dataset
    # Also lowers and strips punctuation
    vectorizer = TextVectorization(max_tokens=7500, output_sequence_length=200)
    text_ds = tf.data.Dataset.from_tensor_slices(df_train['text']).batch(32)
    vectorizer.adapt(text_ds)

    return vectorizer


def get_embedding(vectorizer):
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(2, len(voc))))

    # Load pretrained embeddings
    glove_file = "glove.twitter.27B.50d.txt"

    # Parse embeddings
    embeddings_index = {}
    with open(glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    # 2 extra tokens for padding and unkown words
    num_tokens = len(voc) + 2
    embedding_dim = 50
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word.decode("utf-8"))
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return Embedding(
                num_tokens,
                embedding_dim,
                embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                trainable=False,
            )

def get_model(embedding_layer):
    inputs = Input(shape=(None,), dtype="int64")
    # Embed input in a 50d vector
    x = embedding_layer(inputs)
    # Add 2 bidirectional LSTMs
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(rate = 0.5)(x)
    # Add a classifier
    outputs = Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model



if __name__ == "__main__":
    df_train, df_test = load_dataset()

    # Clean everything included in p.set_options
    df_train['text'] = df_train.apply(lambda x: tweet_tokenizer.tokenize(x.text), axis = 1)
    df_test['text'] = df_test.apply(lambda x: tweet_tokenizer.tokenize(x.text), axis = 1)

    vectorizer = get_vectorizer(df_train, df_test)
    embedding_layer = get_embedding(vectorizer)
    model = get_model(embedding_layer)

    df_train, df_val = train_test_split(df_train, test_size = 0.2)

    x_train = vectorizer(df_train['text'].to_numpy()[..., np.newaxis]).numpy()
    x_val = vectorizer(df_val['text'].to_numpy()[..., np.newaxis]).numpy()

    y_train = df_train['target'].to_numpy()
    y_val = df_val['target'].to_numpy()
    
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))

