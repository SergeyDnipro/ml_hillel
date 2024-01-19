from typing import List, Tuple

import numpy as np
import pandas as pd


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Compute Euclidean distance between two points.

    Args:
        point1 (np.ndarray): First point.
        point2 (np.ndarray): Second point.

    Returns:
        float: Euclidean distance between two points.
    """
    return np.sqrt(np.sum(np.power(point1 - point2, 2)))


class KNN:
    """K Nearest Neighbors classifier."""

    def __init__(self, k: int) -> None:
        """Initialize KNN with the number of neighbors to consider (k).

        Args:
            k (int): Number of neighbors to consider.
        """
        self._X_train = None
        self._y_train = None
        self.k = k  # number of neighbors to consider

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the KNN model with training data.

        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data target.
        """
        self._X_train = X_train
        self._y_train = y_train

    def predict(self, X_test: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Predict target values for test data.

        Args:
            X_test (np.ndarray): Test data features.
            verbose (bool, optional): Print progress during prediction. Defaults to False.

        Returns:
            np.ndarray: Predicted target values.
        """
        n = X_test.shape[0]
        y_pred = np.empty(n, dtype=self._y_train.dtype)

        for i in range(n):
            distances = np.array([euclidean_distance(x, X_test[i]) for x in self._X_train])
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self._y_train[k_indices]
            y_pred[i] = np.bincount(k_nearest_labels).argmax()

            if verbose:
                print(f"Predicted {i + 1}/{n} samples", end="\r")

        if verbose:
            print("")
        return y_pred


def kfold_cross_validation(X: np.ndarray, y: np.ndarray, k: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Split dataset into k folds for cross-validation.

    Args:
        X (np.ndarray): Dataset features.
        y (np.ndarray): Dataset target.
        k (int): Number of folds.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]: List of tuples (X_train, y_train, X_test, y_test).
    """

    n_samples = X.shape[0]
    fold_size = n_samples // k
    folds = []  # Container to store the results of each fold
    for i in range(k):
        cnt = 0
        for del_item in range(0, n_samples, fold_size):
            if cnt == i:
                test_y_target = y[del_item:del_item + fold_size]
                test_X_feature = X[del_item:del_item + fold_size]
                sliced_items = slice(del_item, del_item + fold_size)
                train_y_target = np.concatenate((y[:sliced_items.start], y[sliced_items.stop:]))
                train_X_feature = np.concatenate((X[:sliced_items.start], X[sliced_items.stop:]))
                folds.append((train_X_feature, train_y_target, test_X_feature, test_y_target))
                break
            cnt += 1

    # TODO: implement kfold split. Hints: use `for` Python loop and list slicing.
    return folds


def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: Accuracy score.
    """
    positive = 0
    n = y_true.shape[0]
    for item in range(n):
        positive += 1 if y_true[item] == y_pred[item] else 0

    score = positive/n * 100
    # TODO: implement accuracy score
    return score


def main() -> None:
    """Main function to demonstrate the KNN classifier and k-fold cross-validation."""
    # Read training and testing data from CSV files
    # NOTE: data path, note that it must be specified relative to the \
    # directory from which you run this Python script
    training_data = pd.read_csv("data/train.csv")[:1000]
    testing_data = pd.read_csv("data/test.csv")

    # Extract features and target from the training data
    X = training_data.iloc[:, 1:].values
    y = training_data.iloc[:, 0].values
    print("Training data:", X.shape, y.shape)

    # Extract features and target from the testing data
    X_test = testing_data.iloc[:, 1:].values
    y_test = testing_data.iloc[:, 0].values
    print("Test data:", X_test.shape, y_test.shape)

    k = 5  # NOTE: not the best choice for k
    print(f" KNN with k = {k}")

    num_folds = 5
    summary_accuracy = 0
    # Perform k-fold cross-validation
    for X_train, y_train, X_val, y_val in kfold_cross_validation(X_test, y_test, k=num_folds):
        model = KNN(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val, verbose=True)
        accuracy = evaluate_accuracy(y_val, y_pred)
        summary_accuracy += accuracy
        print(f"Accuracy: {round(accuracy, 2)}")
    print(f"Accuracy per dataset: {round(summary_accuracy / num_folds, 2)}")
    # TODO: compute accuracy on test data and compare results with cross-validation scores


if __name__ == "__main__":
    main()
