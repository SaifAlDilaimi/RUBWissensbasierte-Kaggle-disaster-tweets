import tweet_tokenizer
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import string
import os
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from wordcloud import STOPWORDS
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Embedding, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

nb_epochs = 50

def load_dataset():
    df_train = pd.read_csv('train.csv', dtype={'id': np.int16, 'target': np.int8})
    df_test = pd.read_csv('test.csv', dtype={'id': np.int16})

    print('Training Set Shape = {}'.format(df_train.shape))
    print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
    print('Test Set Shape = {}'.format(df_test.shape))
    print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))

    return df_train, df_test

def prepare_data_for_diagrams(df_train, df_test):
    # word_count
    df_train['word_count'] = df_train['text'].apply(lambda x: len(str(x).split()))
    df_test['word_count'] = df_test['text'].apply(lambda x: len(str(x).split()))

    # unique_word_count
    df_train['unique_word_count'] = df_train['text'].apply(lambda x: len(set(str(x).split())))
    df_test['unique_word_count'] = df_test['text'].apply(lambda x: len(set(str(x).split())))

    # stop_word_count
    df_train['stop_word_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    df_test['stop_word_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

    # url_count
    df_train['url_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
    df_test['url_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

    # mean_word_length
    df_train['mean_word_length'] = df_train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df_test['mean_word_length'] = df_test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    # char_count
    df_train['char_count'] = df_train['text'].apply(lambda x: len(str(x)))
    df_test['char_count'] = df_test['text'].apply(lambda x: len(str(x)))

    # punctuation_count
    df_train['punctuation_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df_test['punctuation_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    # hashtag_count
    df_train['hashtag_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
    df_test['hashtag_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

    # mention_count
    df_train['mention_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
    df_test['mention_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

def plot_distributions(df_train, df_test):
    target_dist_plot = sns.countplot(x = "target", data=df_train)
    fig = target_dist_plot.get_figure()
    fig.savefig("plots/target_dist.png")

    METAFEATURES = ['word_count', 'unique_word_count', 'url_count', 'mean_word_length',
                'char_count', 'punctuation_count', 'hashtag_count', 'mention_count']
    DISASTER_TWEETS = df_train['target'] == 1

    fig, axes = plt.subplots(ncols=2, nrows=len(METAFEATURES), figsize=(20, 50), dpi=100)

    for i, feature in enumerate(METAFEATURES):
        sns.distplot(df_train.loc[~DISASTER_TWEETS][feature], label='Not Disaster', ax=axes[i][0], color='red')
        sns.distplot(df_train.loc[DISASTER_TWEETS][feature], label='Disaster', ax=axes[i][0], color='blue')

        sns.distplot(df_train[feature], label='Training', ax=axes[i][1], color='red')
        sns.distplot(df_test[feature], label='Test', ax=axes[i][1], color='blue')
        
        for j in range(2):
            axes[i][j].set_xlabel('')
            axes[i][j].tick_params(axis='x', labelsize=12)
            axes[i][j].tick_params(axis='y', labelsize=12)
            axes[i][j].legend()
        
        axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)
        axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)

        extent_ax0 = axes[i][0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        extent_ax1 = axes[i][1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'plots/{feature} Target Distribution in Training Set.png', bbox_inches=extent_ax0.expanded(1.1, 1.2))
        fig.savefig(f'plots/{feature} TrainingTest Set Distribution.png', bbox_inches=extent_ax1.expanded(1.1, 1.2))

    #plt.show()

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
    with open(glove_file, encoding="utf8") as f:
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

    tsne_plot(word_index, embeddings_index)

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
    x = Bidirectional(LSTM(8))(x)
    #x = Dropout(rate = 0.2)(x)
    #x = BatchNormalization()(x)
    # Add a classifier
    outputs = Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model

def train():
    df_train, df_test = load_dataset()
    
    prepare_data_for_diagrams(df_train, df_test)
    plot_distributions(df_train, df_test)

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
    
    checkpoint = ModelCheckpoint(filepath="model.h5", 
                                monitor="val_loss", 
                                mode="min", 
                                verbose=1, 
                                save_best_only=True, 
                                save_weights_only=False)

    target_count = df_train['target'].value_counts()
    class_weight = {0: 1.,
                    1: target_count[0] / target_count[1]}

    tensorboard_callback = tf.keras.callbacks.TensorBoard()

    history = model.fit(x_train, y_train, batch_size=32, epochs=nb_epochs, validation_data=(x_val, y_val), callbacks=[checkpoint, tensorboard_callback], class_weight=class_weight)

    save_history(history)

def save_history(history):
    np.save('history.npy', history.history)

def load_history():
    history = np.load('history.npy', allow_pickle='TRUE').item()
    return history

def plot_history():
    history = load_history()
    print(history)

    # training ergebnisse
    loss_training = history['loss']
    acc_training = history['accuracy']
    loss_val = history['val_loss']
    acc_val = history['val_accuracy']

    # plotte ergebnisse
    epochs = range(nb_epochs)
    plt.plot(epochs, loss_training, label="training loss")
    plt.plot(epochs, loss_val, label="validation loss")
    plt.xticks(epochs)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('loss.png')
    plt.clf()

    plt.plot(epochs, acc_training, label="training accuracy")
    plt.plot(epochs, acc_val, label="validation accuracy")
    plt.xticks(epochs)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig('acc.png')

def tsne_plot(word_index, embeddings_index):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in list(word_index.keys())[:500]:
        embedding_vector = embeddings_index.get(word.decode("utf-8"))
        if embedding_vector is not None and len(embedding_vector) == 50:
            tokens.append(embedding_vector)
            labels.append(word.decode("utf-8"))

    tokens_array = np.empty((len(tokens), 50))
    for i in range(len(tokens)):
        tokens_array[i] = tokens[i]

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens_array)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig('test.png')

def main():
    if os.path.exists('model.h5'):
        plot_history()
    else:
        train()

if __name__ == "__main__":
    main()


