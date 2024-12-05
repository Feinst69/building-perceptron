import matplotlib.pyplot as plt
import numpy as np

def plot_loss(losses):
    plt.plot(losses)
    plt.title('Loss over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000, stop_loss=0.01):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.stop_loss = stop_loss
        self.losses = []

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def normalize(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def fit(self, X, y):
        X = self.normalize(X)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = []
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred_i = self.activation_function(linear_output)
                y_pred.append(y_pred_i)
                update = self.learning_rate * (y[idx] - y_pred_i)
                self.weights += update * x_i
                self.bias += update

            y_pred = np.array(y_pred)
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            print(f"Iteration {_}, Loss: {loss}")

            if loss <= self.stop_loss:
                print(f"Stopping early at iteration {_} with loss: {loss}")
                break

    def predict(self, X):
        X = self._normalize(X)
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._activation_function(linear_output)
        return y_pred