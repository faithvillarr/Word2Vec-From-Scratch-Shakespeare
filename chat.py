import numpy as np
from collections import defaultdict
import random

import re

# Step 1: Preprocess text
def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = text.split()
    
    # Create vocabulary and mapping
    vocab = list(set(text))
    word2index = {word: idx for idx, word in enumerate(vocab)}
    index2word = {idx: word for word, idx in word2index.items()}
    
    return text, vocab, word2index, index2word


# Step 2: Create context-target pairs (Skip-gram)
def generate_training_data(text, word2index, window_size=2):
    training_data = []
    for i, word in enumerate(text):
        target_word_idx = word2index[word]
        context_window = text[max(0, i-window_size): min(len(text), i+window_size+1)]
        context_word_indices = [word2index[w] for w in context_window if w != word]
        for context_word_idx in context_word_indices:
            training_data.append((target_word_idx, context_word_idx))
    return np.array(training_data)

# Step 3: Initialize weight matrices
def initialize_weights(vocab_size, embedding_dim):
    W1 = np.random.rand(vocab_size, embedding_dim)
    W2 = np.random.rand(embedding_dim, vocab_size)
    return W1, W2

# Step 4: Define softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Step 5: Train using skip-gram and gradient descent
def train_skipgram(training_data, vocab_size, embedding_dim, epochs, learning_rate):
    W1, W2 = initialize_weights(vocab_size, embedding_dim)
    
    for epoch in range(epochs):
        loss = 0
        for target, context in training_data:
            # Forward pass
            h = W1[target]       # Input to hidden layer
            u = np.dot(W2.T, h)  # Hidden to output layer
            y_pred = softmax(u)

            # Compute error
            e = y_pred
            e[context] -= 1  # One-hot encoded error for correct word

            # Backpropagation
            dW2 = np.outer(h, e)
            dW1 = np.dot(W2, e)

            # Update weights
            W1[target] -= learning_rate * dW1
            W2 -= learning_rate * dW2

            # Apply clipping to avoid NaNs in log loss
            y_pred = np.clip(y_pred, 1e-10, 1.0)

            # Loss (optional, for monitoring)
            loss -= np.log(y_pred[context])
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

    return W1  # W1 contains word embeddings

# Sample usage:
text = "TikTok, whose mainland Chinese counterpart is Douyin, \
    is a short-form video hosting service owned by Chinese internet company ByteDance. \
    It hosts user-submitted videos, which can range in duration from three seconds to 60 minutes. \
    It can be accessed with a smart phone app. \
    Since its launch, TikTok has become one of the world's most popular social media platforms, \
    using recommendation algorithms to connect content creators with new audiences. \
    In April 2020, TikTok surpassed two billion mobile downloads worldwide.\
    Cloudflare ranked TikTok the most popular website of 2021, surpassing Google. \
    The popularity of TikTok has allowed viral trends in food and music to take off \
    and increase the platform's cultural impact worldwide."
preprocessed_text, vocab, word2index, index2word = preprocess_text(text)
training_data = generate_training_data(preprocessed_text, word2index)

# Train the model
embedding_dim = 50
epochs = 1000
learning_rate = 0.01
W1 = train_skipgram(training_data, len(vocab), embedding_dim, epochs, learning_rate)

# Extract word embeddings
word_embeddings = {word: W1[word2index[word]] for word in vocab}

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Apply PCA to reduce dimensionality to 2D
def plot_word_embeddings(word_embeddings, word2index):
    # Extract the word vectors
    word_vectors = np.array([word_embeddings[word] for word in word2index.keys()])
    
    # Perform PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(word_vectors)

    # Step 2: Plot the words
    plt.figure(figsize=(10, 10))
    for word, (x, y) in zip(word2index.keys(), reduced_embeddings):
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word, fontsize=12)
    
    plt.title("Word2Vec Word Embeddings Visualization")
    plt.show()

# Call the function to plot the embeddings
plot_word_embeddings(word_embeddings, word2index)

