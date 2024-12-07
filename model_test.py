if __name__ == "__main__":
    # Imports
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from perceptron import Perceptron, accuracy, load_data, split_data, train_perceptron, evaluate_model

    # Training variables
    learning_rate = 0.01
    n_iter = 1000
    stop_loss = 0.04

    # Load data
    data1 = load_data('forward_selected_features.csv')
    data2 = load_data('backward_selected_features.csv')

    # Split data into features and target
    X1, y1 = split_data(data1, 'diagnosis')
    X2, y2 = split_data(data2, 'diagnosis')


    # Split data1 into training and testing sets
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

    # Initialize and train the perceptron model
    model1 = Perceptron(learning_rate, n_iter)
    model1.fit(X_train1, y_train1, stop_loss)

    model2 = Perceptron(learning_rate, n_iter)
    model2.fit(X_train2, y_train2, stop_loss)

    # Predict and evaluate the model
    y_pred1 = model1.predict(X_test1)
    y_pred2 = model2.predict(X_test2)
    print("Perceptron classification accuracy", accuracy(y_test1, y_pred1))
    print("Perceptron classification accuracy", accuracy(y_test2, y_pred2))

