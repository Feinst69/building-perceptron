import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

# Load the data
data = pd.read_csv('selected_features.csv')

# Split the data into features and target
X = data.drop('diagnosis', axis=1).values
y = data['diagnosis'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the perceptron model
model = Perceptron(learning_rate=0.01, n_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy}")
