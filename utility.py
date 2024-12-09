import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

def load_data(file_path: str):
    return pd.read_csv(file_path)

def split_data(df: pd.DataFrame,
               target_col: str):
    
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values
    return X, y

def train_perceptron(X_train,
                     y_train,
                     eta0=0.1,
                     max_iter=1000,
                     stop_loss=0.02):
    
    model = Perceptron(learning_rate=eta0, n_iter=max_iter)
    model.fit(X_train, y_train,stop_loss)
    return model

def evaluate_model(model,
                   X_test, 
                   y_test):
    
    y_pred = model.predict(X_test)
    return accuracy(y_test, y_pred)


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.losses = []

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def normalize(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def fit(self, X, y, stop_loss=0.02):
        X = self.normalize(X)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = []
            for x_i, y_i in zip(X,y):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred_i = self.activation_function(linear_output)
                y_pred.append(y_pred_i)
                update = self.learning_rate * (y_i - y_pred_i)
                self.weights += update * x_i
                self.bias += update

            y_pred = np.array(y_pred)
            loss = self.compute_loss(y, y_pred)
            if loss <= stop_loss:
                self.losses.append(loss)
                print(f"Converged at iteration {_}; Loss: {loss}")
                break
            else:
                self.losses.append(loss)
                print(f"\rIteration {_}, Loss: {loss}",end="")

        plt.plot(self.losses)
        plt.axhline(y=stop_loss, color='r', linestyle='--' , label='Stop Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Converging of Model')
        plt.legend()
        plt.show()
        plt.savefig(f'Converging of Model.png')

    def predict(self, X):
        X = self.normalize(X)
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_function(linear_output)
        return y_pred