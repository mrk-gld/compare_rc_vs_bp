import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def read_imdb_data(folder_path):    
    data, labels = [], []
    for sentiment in ['pos', 'neg']:
        folder = os.path.join(folder_path, sentiment)
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), 'r') as f:
                data.append(f.read())
                labels.append(1 if sentiment == 'pos' else 0)
    return data, labels#


def preprocess_imdb_data(maxlen, max_words, seed):
    data, labels = read_imdb_data("./aclImdb/train")
    df1 = pd.DataFrame({'review': data, 'label': labels})
    data, labels = read_imdb_data("./aclImdb/test")
    df2 = pd.DataFrame({'review': data, 'label': labels})
    df2 = df2.sample(25_000, random_state=seed)
    df1 = df1.sample(25_000, random_state=seed)
    #combine df1 and df2 into a single dataframe
    df = pd.concat([df1, df2], ignore_index=True)

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df['review'])
    
    embeddings_index = {}
    with open('glove.6B/glove.6B.100d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Prepare embedding matrix
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100  # same as GloVe
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    sequences = tokenizer.texts_to_sequences(df1['review'])
    X = pad_sequences(sequences, maxlen=maxlen)
    y = df1['label'].values

    # Train-Test Split
    X_train, y_train = X.copy(), y.copy()
    
    sequences = tokenizer.texts_to_sequences(df2['review'])
    X = pad_sequences(sequences, maxlen=maxlen)
    y = df2['label'].values
    
    X_test, y_test = X.copy(), y.copy()
    
    return X_train, X_test, y_train, y_test, embedding_matrix