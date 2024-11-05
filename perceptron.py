import numpy as np
import random

# Création de la classe perceptron

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def _activation_function(self, x):
        # Fonction seuil : renvoie 1 si x >= 0, sinon renvoie 0
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        # Initialisation des poids et du biais
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Entraînement du perceptron
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                # Calcul de la prédiction
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self._activation_function(linear_output)

                # Calcul de l'erreur
                error = y[idx] - y_pred

                # Mise à jour des poids et du biais
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error

    def predict(self, X):
        # Prédire les valeurs
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._activation_function(linear_output)
        return y_pred
    
# Générer des données factices
np.random.seed(42)
n_samples = 100

# Générer deux groupes de points pour deux classes
X = np.random.randn(n_samples, 2)
y = np.array([1 if x[0] + x[1] > 0 else 0 for x in X])

# Initialiser et entraîner le perceptron
perceptron = Perceptron(learning_rate=0.1, n_iter=1000)
perceptron.fit(X, y)

# Prédire les étiquettes des données d'entraînement
y_pred = perceptron.predict(X)

# Calcul de l'exactitude (accuracy)
accuracy = np.mean(y_pred == y)
print(f"Exactitude du perceptron sur les données d'entraînement : {accuracy * 100:.2f}%")
