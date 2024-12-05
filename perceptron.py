import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.losses = []

    def _activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def _compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def _normalize(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def fit(self, X, y):
        X = self._normalize(X)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = []
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred_i = self._activation_function(linear_output)
                y_pred.append(y_pred_i)
                update = self.learning_rate * (y[idx] - y_pred_i)
                self.weights += update * x_i
                self.bias += update

            y_pred = np.array(y_pred)
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)
            print(f"Iteration {_}, Loss: {loss}")

    def predict(self, X):
        X = self._normalize(X)
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._activation_function(linear_output)
        return y_pred