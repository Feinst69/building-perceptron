import numpy as np
import pandas as pd
from perceptron2 import artificial_neuron

df = pd.read_csv('forward_selected_features.csv')
print(df.columns)
X = df.drop('diagnosis', axis=1).values
y = df.pop('diagnosis').values


# Train perceptron
W, b = artificial_neuron(X, y, learning_rate=0.01, n_iter=1000)

# # Evaluate perceptron
# y_pred = (model(X_test, W, b) >= 0.5).astype(int)
# accuracy = np.mean(y_pred == y_test)
# print(f"Accuracy: {accuracy * 100:.2f}%")