from perceptron import Perceptron
import numpy as np

if __name__ == "__main__":
    # Generate some dummy data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.array([1 if x[0] + x[1] > 0 else 0 for x in X])

    # Initialize and train the perceptron
    perceptron = Perceptron(learning_rate=0.01, n_iter=10)
    perceptron.fit(X, y)

    # Predict the values
    predictions = perceptron.predict(X)
    print(y)
    print(predictions)