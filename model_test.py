if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from utility import load_data, split_data, train_perceptron, evaluate_model
    import matplotlib.pyplot as plt

    # Training variables
    learning_rate = 5
    n_iter = 1000
    stop_loss = 0.03

    # Load data
    data1 = load_data('forward_selected_features.csv')
    data2 = load_data('backward_selected_features.csv')

    # Split data into features and target
    X1, y1 = split_data(data1, 'diagnosis')
    X2, y2 = split_data(data2, 'diagnosis')


    # Split data1 into training and testing sets
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

    # Train the perceptron models
    model1 = train_perceptron(X_train1, y_train1, learning_rate, n_iter, stop_loss)
    model2 = train_perceptron(X_train2, y_train2, learning_rate, n_iter, stop_loss)

    # Evaluate the models
    accuracy1 = evaluate_model(model1, X_test1, y_test1)
    accuracy2 = evaluate_model(model2, X_test2, y_test2)

    print("Perceptron classification accuracy (forward selected features):", accuracy1)
    print("Perceptron classification accuracy (backward selected features):", accuracy2)

