if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from utility import load_data, split_data, train_perceptron, evaluate_model
    import matplotlib.pyplot as plt

    # Training variables
    learning_rate = 0.1
    n_iter = 1000
    stop_loss = 0.035

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
    acc1, precision1, recall1, f1_1 = evaluate_model(model1, X_test1, y_test1)
    acc2, precision2, recall2, f1_2 = evaluate_model(model2, X_test2, y_test2)

    print("Perceptron classification accuracy (forward selected features):", acc1)
    print("Precision (forward selected features):", precision1)
    print("Recall (forward selected features):", recall1)
    print("F1-score (forward selected features):", f1_1)

    print("Perceptron classification accuracy (backward selected features):", acc2)
    print("Precision (backward selected features):", precision2)
    print("Recall (backward selected features):", recall2)
    print("F1-score (backward selected features):", f1_2)

