
import csv
import numpy as np


def load_dataset_and_split(path="data/records.csv"):

    # Load dataset
    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        data = list(reader)

    predictor_columns = [
        "experience_months",
        "interview_reschedules_90d",
        "salary_gap_ratio",
        "expected_salary_inr_thousands",
        "profile_completeness_score",
        "competing_offer_flag",
        "referred_flag"
    ]

    X_all = np.array([
        [float(row[column]) for column in predictor_columns]
        for row in data
    ])

    y_all = np.array([
        int(row["offer_declined"])
        for row in data
    ])

    # Stratified 70/30 split
    class_0_indices = np.where(y_all == 0)[0]
    class_1_indices = np.where(y_all == 1)[0]

    class_0_train_count = (7 * len(class_0_indices)) // 10
    class_1_train_count = (7 * len(class_1_indices)) // 10

    train_indices = sorted(
        class_0_indices[:class_0_train_count].tolist()
        + class_1_indices[:class_1_train_count].tolist()
    )

    test_indices = sorted(
        class_0_indices[class_0_train_count:].tolist()
        + class_1_indices[class_1_train_count:].tolist()
    )

    # Create training and testing arrays
    X_train = X_all[train_indices]
    X_test = X_all[test_indices]

    y_train = y_all[train_indices]
    y_test = y_all[test_indices]

    # Standardization using training rows only
    train_mean = np.mean(X_train, axis=0)
    train_sd = np.std(X_train, axis=0, ddof=1)

    X_train_scaled = (X_train - train_mean) / train_sd
    X_test_scaled = (X_test - train_mean) / train_sd

    return {
        "data": data,
        "X_all": X_all,
        "y_all": y_all,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "train_mean": train_mean,
        "train_sd": train_sd
    }

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y, probabilities):
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return -np.mean(
        y * np.log(probabilities)
        + (1 - y) * np.log(1 - probabilities)
    )


def fit_logistic_regression(
    X,
    y,
    learning_rate=0.1,
    epochs=1000,
    return_history=False,
):
    """Fit logistic regression; optionally return required loss checkpoints."""
    weights = np.zeros(X.shape[1])
    bias = 0.0
    history = {}

    checkpoints = {0, 1, 10, 100, epochs}
    if return_history:
        initial_probabilities = sigmoid(X @ weights + bias)
        history[0] = binary_cross_entropy(y, initial_probabilities)

    for epoch in range(1, epochs + 1):
        probabilities = sigmoid(X @ weights + bias)

        dw = X.T @ (probabilities - y) / len(y)
        db = np.mean(probabilities - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

        if return_history and epoch in checkpoints:
            checkpoint_probabilities = sigmoid(X @ weights + bias)
            history[epoch] = binary_cross_entropy(y, checkpoint_probabilities)

    if return_history:
        return weights, bias, history

    return weights, bias


def knn_probabilities(X_reference, y_reference, X_query, k):
    probabilities = []

    for query in X_query:
        distances = np.sqrt(
            np.sum((X_reference - query) ** 2, axis=1)
        )

        nearest_indices = np.argsort(
            distances, kind="stable"
        )[:k]

        positive_count = np.sum(y_reference[nearest_indices] == 1)
        probabilities.append(positive_count / k)

    return np.array(probabilities)


def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )


def select_best_knn_k(X, y):
    candidate_ks = range(1, 22, 2)
    folds = np.arange(len(y)) % 5
    cv_results = []

    for k in candidate_ks:
        fold_f1_scores = []

        for fold in range(5):
            validation_indices = np.where(folds == fold)[0]
            training_indices = np.where(folds != fold)[0]

            X_fold_train = X[training_indices]
            y_fold_train = y[training_indices]
            X_fold_validation = X[validation_indices]
            y_fold_validation = y[validation_indices]

            probabilities = knn_probabilities(
                X_fold_train,
                y_fold_train,
                X_fold_validation,
                k
            )

            predictions = (
                probabilities >= (k + 1) / 2 / k
            ).astype(int)

            fold_f1_scores.append(
                f1_score(y_fold_validation, predictions)
            )

        mean_f1 = np.mean(fold_f1_scores)

        cv_results.append({
            "K": k,
            "mean_cv_f1": mean_f1
        })

    best_result = max(
        cv_results,
        key=lambda row: row["mean_cv_f1"]
    )

    return best_result["K"], cv_results
