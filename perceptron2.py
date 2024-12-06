import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, stop_loss=None):
        self.weights = np.zeros(input_size)
        self.learning_rate = learning_rate
        self.stop_loss = stop_loss
        self.loss_history = []

    def predict(self, inputs):
        summation = np.dot(self.weights, inputs)
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels, epochs=1000):
        training_inputs = np.array(training_inputs)
        labels = np.array(labels)
        
        for epoch in range(epochs):
            total_loss = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                total_loss += abs(error)
                self.weights += self.learning_rate * error * inputs
            self.loss_history.append(total_loss)
            if self.stop_loss is not None and total_loss <= self.stop_loss:
                print(f"Training stopped at epoch {epoch} due to stop loss condition.")
                break

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Loss Evolution')
        plt.show()