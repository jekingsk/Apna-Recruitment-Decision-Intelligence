import csv
import os
import numpy as np

from ml_helpers import (
    load_dataset_and_split,
    sigmoid,
    binary_cross_entropy,
    fit_logistic_regression,
    knn_probabilities,
    f1_score,
    select_best_knn_k,
)


# ============================================================
# COMMON DATA PREPARATION - TASK 2.1
# ============================================================

prepared = load_dataset_and_split()

data = prepared["data"]
X_all = prepared["X_all"]
y_all = prepared["y_all"]
train_indices = prepared["train_indices"]
test_indices = prepared["test_indices"]

X_train = prepared["X_train"]
X_test = prepared["X_test"]
y_train = prepared["y_train"]
y_test = prepared["y_test"]

X_train_scaled = prepared["X_train_scaled"]
X_test_scaled = prepared["X_test_scaled"]

os.makedirs("outputs", exist_ok=True)

results = {
    "train_rows": len(X_train),
    "train_positives": int(np.sum(y_train)),
    "test_rows": len(X_test),
    "test_positives": int(np.sum(y_test)),
}


def save_results():
    """Rewrite the complete Part 2 results file."""
    with open("outputs/part2_results.txt", "w", encoding="utf-8") as file:
        for key, value in results.items():
            if isinstance(value, (float, np.floating)):
                file.write(f"{key} = {value:.4f}\n")
            else:
                file.write(f"{key} = {value}\n")


def display_results(title, keys=None):
    print(f"\n{title}")
    selected_keys = keys if keys is not None else list(results.keys())

    for key in selected_keys:
        value = results[key]
        if isinstance(value, (float, np.floating)):
            print(f"{key} = {value:.4f}")
        else:
            print(f"{key} = {value}")


# Save and display Task 2.1
save_results()
display_results("Task 2.1 - Stratified Train/Test Split")


# ============================================================
# TASK 2.2 - LOGISTIC REGRESSION FROM SCRATCH
# ============================================================

learning_rate = 0.1
epochs = 1000

weights, bias, loss_history = fit_logistic_regression(
    X_train_scaled,
    y_train,
    learning_rate=learning_rate,
    epochs=epochs,
    return_history=True,
)

train_probabilities = sigmoid(X_train_scaled @ weights + bias)
test_probabilities = sigmoid(X_test_scaled @ weights + bias)
test_loss = binary_cross_entropy(y_test, test_probabilities)

for epoch in [0, 1, 10, 100, 1000]:
    results[f"loss_epoch_{epoch}"] = loss_history[epoch]

for i, weight in enumerate(weights, start=1):
    results[f"weight_X{i}"] = weight

results["bias"] = bias
results["test_bce"] = test_loss

save_results()

display_results(
    "Task 2.2 - Logistic Regression",
    [
        "loss_epoch_0",
        "loss_epoch_1",
        "loss_epoch_10",
        "loss_epoch_100",
        "loss_epoch_1000",
        "bias",
        "weight_X1",
        "weight_X2",
        "weight_X3",
        "weight_X4",
        "weight_X5",
        "weight_X6",
        "weight_X7",
        "test_bce",
    ],
)


# ============================================================
# TASK 2.3 - LEARNING RATE COMPARISON
# ============================================================

learning_rates = [0.01, 0.1, 1.0]
comparison_results = {}

for eta in learning_rates:
    # Restart weights and bias from zero for each required learning rate.
    trial_weights = np.zeros(X_train_scaled.shape[1])
    trial_bias = 0.0

    for _ in range(200):
        probabilities = sigmoid(
            X_train_scaled @ trial_weights + trial_bias
        )
        error = probabilities - y_train

        dw = (X_train_scaled.T @ error) / len(y_train)
        db = np.mean(error)

        trial_weights -= eta * dw
        trial_bias -= eta * db

    final_probabilities = sigmoid(
        X_train_scaled @ trial_weights + trial_bias
    )
    final_loss = binary_cross_entropy(y_train, final_probabilities)

    comparison_results[eta] = final_loss
    results[f"loss_epoch_200_eta_{eta}"] = final_loss

save_results()

print("\nTask 2.3 - Learning Rate Comparison")
for eta, loss in comparison_results.items():
    print(f"loss_epoch_200_eta_{eta} = {loss:.4f}")


# ============================================================
# TASK 2.4 - ODDS RATIOS
# ============================================================

odds_ratios = np.exp(weights)

for i, odds_ratio in enumerate(odds_ratios, start=1):
    results[f"odds_ratio_X{i}"] = odds_ratio

save_results()

print("\nTask 2.4 - Odds Ratios")
for i, odds_ratio in enumerate(odds_ratios, start=1):
    print(f"odds_ratio_X{i} = {odds_ratio:.4f}")


# ============================================================
# TASK 2.5 - TEST EVALUATION AT THRESHOLDS 0.5 AND 0.3
# ============================================================

def evaluate_threshold(y_actual, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tp = int(np.sum((y_actual == 1) & (predictions == 1)))
    fp = int(np.sum((y_actual == 0) & (predictions == 1)))
    tn = int(np.sum((y_actual == 0) & (predictions == 0)))
    fn = int(np.sum((y_actual == 1) & (predictions == 0)))

    accuracy = (tp + tn) / len(y_actual)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


metrics_05 = evaluate_threshold(y_test, test_probabilities, 0.5)
metrics_03 = evaluate_threshold(y_test, test_probabilities, 0.3)

for metric, value in metrics_05.items():
    results[f"tau_0.5_{metric}"] = value

for metric, value in metrics_03.items():
    results[f"tau_0.3_{metric}"] = value

blind_predictions = np.zeros(len(y_test), dtype=int)
results["blind_accuracy"] = np.mean(blind_predictions == y_test)

save_results()

print("\nTask 2.5 - Threshold Evaluation")
for key, value in results.items():
    if key.startswith("tau_") or key == "blind_accuracy":
        if isinstance(value, (float, np.floating)):
            print(f"{key} = {value:.4f}")
        else:
            print(f"{key} = {value}")


# ============================================================
# TASK 2.6 - THRESHOLD SWEEP / ROC DATA
# ============================================================

thresholds = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9,
]

roc_rows = []

for threshold in thresholds:
    predictions = (test_probabilities >= threshold).astype(int)

    tp = int(np.sum((y_test == 1) & (predictions == 1)))
    fp = int(np.sum((y_test == 0) & (predictions == 1)))
    tn = int(np.sum((y_test == 0) & (predictions == 0)))
    fn = int(np.sum((y_test == 1) & (predictions == 0)))

    tpr = tp / (tp + fn) if tp + fn > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0

    roc_rows.append(
        {
            "tau": f"{threshold:.4f}",
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tpr": f"{tpr:.4f}",
            "fpr": f"{fpr:.4f}",
        }
    )

with open(
    "outputs/part2_thresholds.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:
    fieldnames = ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(roc_rows)

print("\nTask 2.6 - Threshold Sweep")
for row in roc_rows:
    print(row)

print("\nSaved: outputs/part2_thresholds.csv")


# ============================================================
# TASK 2.7 - KNN WITH 5-FOLD CROSS-VALIDATION
# ============================================================

best_k, knn_cv_results = select_best_knn_k(
    X_train_scaled,
    y_train,
)

# Save cross-validation results.
with open(
    "outputs/part2_knn_cv.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["K", "mean_cv_f1"],
    )
    writer.writeheader()

    for row in knn_cv_results:
        writer.writerow(
            {
                "K": row["K"],
                "mean_cv_f1": f"{row['mean_cv_f1']:.4f}",
            }
        )

# Evaluate the selected K using all training rows as reference.
knn_test_probabilities = knn_probabilities(
    X_train_scaled,
    y_train,
    X_test_scaled,
    best_k,
)

# Majority-vote threshold; equivalent to the original KNN classifier.
knn_test_predictions = (
    knn_test_probabilities >= ((best_k + 1) / 2) / best_k
).astype(int)

knn_tp = int(np.sum((y_test == 1) & (knn_test_predictions == 1)))
knn_fp = int(np.sum((y_test == 0) & (knn_test_predictions == 1)))
knn_tn = int(np.sum((y_test == 0) & (knn_test_predictions == 0)))
knn_fn = int(np.sum((y_test == 1) & (knn_test_predictions == 0)))

knn_accuracy = np.mean(knn_test_predictions == y_test)
knn_precision = (
    knn_tp / (knn_tp + knn_fp)
    if knn_tp + knn_fp > 0
    else 0.0
)
knn_recall = (
    knn_tp / (knn_tp + knn_fn)
    if knn_tp + knn_fn > 0
    else 0.0
)
knn_f1 = f1_score(y_test, knn_test_predictions)

results["knn_best_k"] = best_k
results.update(
    {
        "knn_tp": knn_tp,
        "knn_fp": knn_fp,
        "knn_tn": knn_tn,
        "knn_fn": knn_fn,
        "knn_accuracy": knn_accuracy,
        "knn_precision": knn_precision,
        "knn_recall": knn_recall,
        "knn_f1": knn_f1,
    }
)

save_results()

print("\nTask 2.7 - KNN Cross-Validation")
for row in knn_cv_results:
    print(
        f"K = {row['K']}, "
        f"mean_cv_f1 = {row['mean_cv_f1']:.4f}"
    )

print(f"\nBest K = {best_k}")

display_results(
    "KNN Test Results",
    [
        "knn_best_k",
        "knn_tp",
        "knn_fp",
        "knn_tn",
        "knn_fn",
        "knn_accuracy",
        "knn_precision",
        "knn_recall",
        "knn_f1",
    ],
)

print("\nSaved: outputs/part2_results.txt")
print("Saved: outputs/part2_knn_cv.csv")


# ==========================================
# TASK 2.8 - COST-BASED THRESHOLD SELECTION
# ==========================================

# Cost assumptions in INR
FALSE_ALARM_COST = 50
MISSED_POSITIVE_COST = 400


# Calculate financial cost for a given threshold.
def calculate_cost(y_actual, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    fp = int(np.sum(
        (y_actual == 0) & (predictions == 1)
    ))

    fn = int(np.sum(
        (y_actual == 1) & (predictions == 0)
    ))

    total_cost = (
        FALSE_ALARM_COST * fp
        + MISSED_POSITIVE_COST * fn
    )

    return total_cost, fp, fn


# ------------------------------------------
# 1. FIND BEST THRESHOLD USING TRAINING DATA
# ------------------------------------------

# Evaluate thresholds: 0.05, 0.10, ..., 0.95
candidate_thresholds = [
    k / 20 for k in range(1, 20)
]

best_cost = float("inf")
cost_tau = None
cost_train_at_tau = None

for threshold in candidate_thresholds:

    training_cost, fp, fn = calculate_cost(
        y_train,
        train_probabilities,
        threshold
    )

    # Strictly less-than preserves the smaller threshold
    # when two thresholds have equal cost.
    if training_cost < best_cost:

        best_cost = training_cost
        cost_tau = threshold
        cost_train_at_tau = training_cost


# ------------------------------------------
# 2. EVALUATE SELECTED THRESHOLD ON TEST DATA
# ------------------------------------------

cost_test_at_tau, test_fp, test_fn = calculate_cost(
    y_test,
    test_probabilities,
    cost_tau
)


# ------------------------------------------
# 3. EVALUATE TEST DATA AT THRESHOLD 0.5
# ------------------------------------------

cost_test_at_0_5, fp_05, fn_05 = calculate_cost(
    y_test,
    test_probabilities,
    0.5
)


# ------------------------------------------
# 4. COST OF FLAGGING NOBODY
# ------------------------------------------

# No records are flagged.
# Therefore, every positive test observation is missed.

cost_test_flag_nobody = (
    MISSED_POSITIVE_COST * np.sum(y_test == 1)
)


# ------------------------------------------
# 5. STORE REQUIRED RESULTS
# ------------------------------------------

results.update({

    "cost_tau": cost_tau,

    "cost_train_at_tau": cost_train_at_tau,

    "cost_test_at_tau": cost_test_at_tau,

    "cost_test_at_0.5": cost_test_at_0_5,

    "cost_test_flag_nobody": cost_test_flag_nobody

})


# ------------------------------------------
# 6. SAVE COMPLETE PART 2 RESULTS
# ------------------------------------------

with open(
    "outputs/part2_results.txt",
    "w",
    encoding="utf-8"
) as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")
        else:
            file.write(f"{key} = {value}\n")


# ------------------------------------------
# 7. DISPLAY RESULTS
# ------------------------------------------

print("\nTask 2.8 - Cost-Based Threshold Selection")

print(f"cost_tau = {cost_tau:.2f}")
print(f"cost_train_at_tau = INR {cost_train_at_tau}")
print(f"cost_test_at_tau = INR {cost_test_at_tau}")
print(f"cost_test_at_0.5 = INR {cost_test_at_0_5}")
print(f"cost_test_flag_nobody = INR {cost_test_flag_nobody}")

print("\nSaved: outputs/part2_results.txt")
