import numpy as np
import pandas as pd
from perceptron2 import artificial_neuron, load_data
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
X, y = load_data('forward_selected_features.csv')

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train perceptron
W, b = artificial_neuron(X_train, y_train, learning_rate=0.01, n_iter=1000)

# # Evaluate perceptron
# y_pred = (model(X_test, W, b) >= 0.5).astype(int)
# accuracy = np.mean(y_pred == y_test)
# print(f"Accuracy: {accuracy * 100:.2f}%")