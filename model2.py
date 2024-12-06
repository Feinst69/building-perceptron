import numpy as np
import pandas as pd
from perceptron2 import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('forward_selected_features.csv')

# Preprocess dataset
X = data.drop('diagnosis', axis=1).values
y = data.pop('diagnosis')
y = y.map({'M': 1, 'B': 0})
print(y)
# y = LabelEncoder().fit_transform(data['diagnosis']).astype(int)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train perceptron
input_size = X_train.shape[1]
perceptron = Perceptron(input_size=input_size, learning_rate=0.01, stop_loss=0.01)
perceptron.train(X_train, y_train, epochs=1000)

# Evaluate perceptron
predictions = [perceptron.predict(x) for x in X_test]
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot loss evolution
perceptron.plot_loss()