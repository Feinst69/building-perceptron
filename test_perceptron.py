from utility import train_perceptron, evaluate_model
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    # Generate some dummy data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.array([1 if x[0] + x[1] > 0 else 0 for x in X])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    # Training variables
    learning_rate = 0.01
    n_iter = 1000
    stop_loss = 0.03

    # Train the perceptron models
    model = train_perceptron(X_train, y_train, learning_rate, n_iter, stop_loss)

    # Evaluate the models
    accuracy = evaluate_model(model, X_test, y_test)

    print("Perceptron classification accuracy (forward selected features):", accuracy)