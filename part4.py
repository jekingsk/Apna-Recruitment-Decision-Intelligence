
import csv
import os
import numpy as np

from ml_helpers import (
    load_dataset_and_split,
    sigmoid,
    fit_logistic_regression,
    knn_probabilities,
    select_best_knn_k
)

# ==========================================
# TASK 4.1 - WHICH SPLIT IS PUREST?
# ==========================================

# Load shared training and testing data
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

# Original, unstandardised training data
X_train_original = X_all[train_indices]

# Entropy function
def entropy(y):

    n = len(y)

    if n == 0:
        return 0.0

    positives = np.sum(y == 1)
    negatives = np.sum(y == 0)

    p1 = positives / n
    p0 = negatives / n

    result = 0.0

    if p0 > 0:
        result -= p0 * np.log2(p0)

    if p1 > 0:
        result -= p1 * np.log2(p1)

    return result


# Gini impurity function
def gini(y):

    n = len(y)

    if n == 0:
        return 0.0

    p1 = np.mean(y == 1)
    p0 = np.mean(y == 0)

    return 1 - (p0 ** 2 + p1 ** 2)


# Parent node impurity
root_entropy = entropy(y_train)
root_gini = gini(y_train)

# Define four candidate splits
split_conditions = {
    "S1": X_train_original[:, 5] == 1,
    "S2": X_train_original[:, 6] == 1,
    "S3": X_train_original[:, 2] > 0.5,
    "S4": X_train_original[:, 0] <= 24
}

split_rows = []

# Evaluate each split
for split_name, yes_mask in split_conditions.items():

    no_mask = ~yes_mask

    y_yes = y_train[yes_mask]
    y_no = y_train[no_mask]

    n_yes = len(y_yes)
    n_no = len(y_no)

    positives_yes = int(np.sum(y_yes == 1))
    positives_no = int(np.sum(y_no == 1))

    # Weighted child entropy
    weighted_entropy = (
        (n_yes / len(y_train)) * entropy(y_yes)
        + (n_no / len(y_train)) * entropy(y_no)
    )

    # Weighted child Gini
    weighted_gini = (
        (n_yes / len(y_train)) * gini(y_yes)
        + (n_no / len(y_train)) * gini(y_no)
    )

    # Information and Gini gain
    info_gain = root_entropy - weighted_entropy
    gini_gain = root_gini - weighted_gini

    split_rows.append({
        "split": split_name,
        "n_yes": n_yes,
        "positives_yes": positives_yes,
        "n_no": n_no,
        "positives_no": positives_no,
        "info_gain": f"{info_gain:.4f}",
        "gini_gain": f"{gini_gain:.4f}"
    })


# Find best split by information gain
best_split_by_info_gain = max(
    split_rows,
    key=lambda row: float(row["info_gain"])
)["split"]

# Find best split by Gini gain
best_split_by_gini_gain = max(
    split_rows,
    key=lambda row: float(row["gini_gain"])
)["split"]


# Store results
results = {
    "root_entropy": root_entropy,
    "root_gini": root_gini,
    "best_split_by_info_gain": best_split_by_info_gain,
    "best_split_by_gini_gain": best_split_by_gini_gain
}


# Save split CSV
os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/part4_splits.csv",
    "w",
    newline=""
) as file:

    fieldnames = [
        "split",
        "n_yes",
        "positives_yes",
        "n_no",
        "positives_no",
        "info_gain",
        "gini_gain"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(split_rows)


# Save results
with open("outputs/part4_results.txt", "w") as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")
        else:
            file.write(f"{key} = {value}\n")


# Display results
print("Task 4.1 - Split Evaluation")

print(f"root_entropy = {root_entropy:.4f}")
print(f"root_gini = {root_gini:.4f}")

for row in split_rows:
    print(row)

print("best_split_by_info_gain =", best_split_by_info_gain)
print("best_split_by_gini_gain =", best_split_by_gini_gain)

print("\nSaved: outputs/part4_splits.csv")


# ==========================================
# TASK 4.2 - DECISION STUMP AND VOTING
# ==========================================

# 1. Train logistic regression using the same
# settings as Part 2, Task 2.2.

log_weights, log_bias = fit_logistic_regression(
    X_train_scaled,
    y_train,
    learning_rate=0.1,
    epochs=1000
)

logistic_probabilities = sigmoid(
    X_test_scaled @ log_weights + log_bias
)

logistic_predictions = (
    logistic_probabilities >= 0.3
).astype(int)


# 2. Find the best K using the same five-fold CV
# procedure as Part 2, Task 2.7.

best_k, knn_cv_results = select_best_knn_k(
    X_train_scaled,
    y_train
)

knn_test_probabilities = knn_probabilities(
    X_train_scaled,
    y_train,
    X_test_scaled,
    best_k
)

# Majority vote: at least (K+1)/2 positive neighbours.
knn_predictions = (
    knn_test_probabilities >= ((best_k + 1) / 2) / best_k
).astype(int)


# 3. Decision stump using the best information-gain split.
# Task 4.1 selected S4: experience_months <= 24.

stump_yes_mask = X_train_original[:, 0] <= 24
stump_no_mask = ~stump_yes_mask

stump_rate_yes = np.mean(y_train[stump_yes_mask])
stump_rate_no = np.mean(y_train[stump_no_mask])

# Apply the same rule to test records.
test_yes_mask = X_test[:, 0] <= 24

stump_probabilities = np.where(
    test_yes_mask,
    stump_rate_yes,
    stump_rate_no
)

stump_predictions = (
    stump_probabilities >= 0.3
).astype(int)


# 4. Hard voting: at least 2 of 3 models predict positive.

hard_vote_count = (
    logistic_predictions
    + knn_predictions
    + stump_predictions
)

hard_vote_predictions = (
    hard_vote_count >= 2
).astype(int)


# 5. Soft voting: average the three model probabilities.

soft_vote_probabilities = (
    logistic_probabilities
    + knn_test_probabilities
    + stump_probabilities
) / 3

soft_vote_predictions = (
    soft_vote_probabilities >= 0.3
).astype(int)


# 6. Evaluation function.

def evaluate_model(y_actual, predictions):
    tp = int(np.sum((y_actual == 1) & (predictions == 1)))
    fp = int(np.sum((y_actual == 0) & (predictions == 1)))
    tn = int(np.sum((y_actual == 0) & (predictions == 0)))
    fn = int(np.sum((y_actual == 1) & (predictions == 0)))

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# 7. Evaluate all five models in the required order.

model_predictions = [
    ("logistic", logistic_predictions),
    ("knn", knn_predictions),
    ("stump", stump_predictions),
    ("hard_vote", hard_vote_predictions),
    ("soft_vote", soft_vote_predictions)
]

voting_rows = []

for model_name, predictions in model_predictions:

    metrics = evaluate_model(y_test, predictions)

    voting_rows.append({
        "model": model_name,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "precision": f"{metrics['precision']:.4f}",
        "recall": f"{metrics['recall']:.4f}",
        "f1": f"{metrics['f1']:.4f}"
    })


# 8. Store required results.

results.update({
    "stump_rate_yes": stump_rate_yes,
    "stump_rate_no": stump_rate_no
})


# 9. Save voting comparison CSV.

with open(
    "outputs/part4_voting.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "model",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(voting_rows)


# 10. Update Part 4 results file.

with open(
    "outputs/part4_results.txt",
    "w",
    encoding="utf-8"
) as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")
        else:
            file.write(f"{key} = {value}\n")


# 11. Display results.

print("\nTask 4.2 - Decision Stump and Voting")

print(f"Best K = {best_k}")
print(f"stump_rate_yes = {stump_rate_yes:.4f}")
print(f"stump_rate_no = {stump_rate_no:.4f}")

for row in voting_rows:
    print(row)

print("\nSaved: outputs/part4_voting.csv")


# ==========================================
# TASK 4.3 - ADABOOST FROM SCRATCH
# ==========================================

# Actual labels and predictions of two weak classifiers.
# Labels: +1 and -1.

y_adaboost = np.array([
    1, 1, 1, -1, -1, -1, 1, -1
])

h1 = np.array([
    1, 1, -1, -1, -1, -1, -1, -1
])

h2 = np.array([
    -1, 1, 1, -1, 1, -1, 1, -1
])

n = len(y_adaboost)

# Initially, every observation has equal weight.
weights = np.full(n, 1 / n, dtype=float)

adaboost_rows = []


# ------------------------------------------
# ROUND 1 - TRAIN CLASSIFIER H1
# ------------------------------------------

# Weighted classification error.
round1_error = np.sum(
    weights[y_adaboost != h1]
)

# Classifier importance.
round1_alpha = 0.5 * np.log(
    (1 - round1_error) / round1_error
)

round1_weights_before = weights.copy()

# Update weights:
# Correct predictions decrease in weight.
# Incorrect predictions increase in weight.

round1_unnormalised = (
    weights * np.exp(-round1_alpha * y_adaboost * h1)
)

round1_weights_after = (
    round1_unnormalised / np.sum(round1_unnormalised)
)

# Store round 1 values.
for i in range(n):
    adaboost_rows.append({
        "round": 1,
        "point": i + 1,
        "weight_before": f"{round1_weights_before[i]:.4f}",
        "weight_unnormalised": f"{round1_unnormalised[i]:.4f}",
        "weight_after": f"{round1_weights_after[i]:.4f}"
    })


# Total normalized weight of observations misclassified
# by the first classifier.
round1_misclassified_weight = np.sum(
    round1_weights_after[y_adaboost != h1]
)


# ------------------------------------------
# ROUND 2 - TRAIN CLASSIFIER H2
# ------------------------------------------

# Use the normalized weights from round 1.
weights = round1_weights_after.copy()

round2_error = np.sum(
    weights[y_adaboost != h2]
)

round2_alpha = 0.5 * np.log(
    (1 - round2_error) / round2_error
)

round2_weights_before = weights.copy()

round2_unnormalised = (
    weights * np.exp(-round2_alpha * y_adaboost * h2)
)

round2_weights_after = (
    round2_unnormalised / np.sum(round2_unnormalised)
)

# Store round 2 values.
for i in range(n):
    adaboost_rows.append({
        "round": 2,
        "point": i + 1,
        "weight_before": f"{round2_weights_before[i]:.4f}",
        "weight_unnormalised": f"{round2_unnormalised[i]:.4f}",
        "weight_after": f"{round2_weights_after[i]:.4f}"
    })


# ------------------------------------------
# FINAL ADABOOST ENSEMBLE
# ------------------------------------------

# Weighted combination of both weak classifiers.
ensemble_scores = (
    round1_alpha * h1
    + round2_alpha * h2
)

# Predict +1 for positive scores, otherwise -1.
ensemble_predictions = np.where(
    ensemble_scores > 0,
    1,
    -1
)

ensemble_accuracy = np.mean(
    ensemble_predictions == y_adaboost
)


# ------------------------------------------
# SAVE REQUIRED RESULTS
# ------------------------------------------

results.update({
    "round1_error": round1_error,
    "round1_alpha": round1_alpha,
    "round2_error": round2_error,
    "round2_alpha": round2_alpha,
    "round1_misclassified_weight": round1_misclassified_weight,
    "ensemble_accuracy": ensemble_accuracy
})


# Save the 16 weight-update rows.
with open(
    "outputs/part4_adaboost.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "round",
        "point",
        "weight_before",
        "weight_unnormalised",
        "weight_after"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(adaboost_rows)


# Update the complete Part 4 results file.
with open(
    "outputs/part4_results.txt",
    "w",
    encoding="utf-8"
) as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")
        else:
            file.write(f"{key} = {value}\n")


# ------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------

print("\nTask 4.3 - AdaBoost")

print(f"round1_error = {round1_error:.4f}")
print(f"round1_alpha = {round1_alpha:.4f}")
print(f"round2_error = {round2_error:.4f}")
print(f"round2_alpha = {round2_alpha:.4f}")

print(
    "round1_misclassified_weight = "
    f"{round1_misclassified_weight:.4f}"
)

print(f"ensemble_accuracy = {ensemble_accuracy:.4f}")

print("\nSaved: outputs/part4_adaboost.csv")


# ==========================================
# TASK 4.4 - OUT-OF-BAG PROBABILITY
# ==========================================

n_train = len(y_train)

# Probability that a particular training observation
# is never selected in n_train draws with replacement.
oob_fraction = (1 - 1 / n_train) ** n_train

# Store result.
results["oob_fraction"] = oob_fraction

# Rewrite complete Part 4 results.
with open(
    "outputs/part4_results.txt",
    "w",
    encoding="utf-8"
) as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")
        else:
            file.write(f"{key} = {value}\n")

print("\nTask 4.4 - Out-of-Bag Probability")
print(f"oob_fraction = {oob_fraction:.4f}")


# ==========================================
# TASK 4.5 - NEURAL NETWORK ARCHITECTURES
# ==========================================

# ------------------------------------------
# PART A - MLP PARAMETER COUNT
# ------------------------------------------

# Architecture: 784 -> 16 -> 16 -> 10

mlp_hidden1 = (784 * 16) + 16
mlp_hidden2 = (16 * 16) + 16
mlp_output = (16 * 10) + 10

mlp_total = (
    mlp_hidden1
    + mlp_hidden2
    + mlp_output
)


# ------------------------------------------
# PART B - CNN PARAMETER COUNT
# ------------------------------------------

# Input image: 28 x 28 x 1

# CONVOLUTION 1
# Filter = 5 x 5
# Input channels = 1
# Number of filters = 6

cnn_conv1 = (5 * 5 * 1 + 1) * 6

# Output shape: 28 x 28 x 6
# Max pooling 2 x 2 gives: 14 x 14 x 6


# CONVOLUTION 2
# Filter = 5 x 5
# Input channels = 6
# Number of filters = 16

cnn_conv2 = (5 * 5 * 6 + 1) * 16

# Input shape: 14 x 14 x 6
# Output shape after convolution: 10 x 10 x 16
# After max pooling: 5 x 5 x 16


# FLATTEN
flattened_features = 5 * 5 * 16

# DENSE LAYER 1
# 400 inputs -> 32 neurons

cnn_dense1 = (flattened_features * 32) + 32

# DENSE LAYER 2
# 32 inputs -> 10 classes

cnn_dense2 = (32 * 10) + 10

cnn_total = (
    cnn_conv1
    + cnn_conv2
    + cnn_dense1
    + cnn_dense2
)


# ------------------------------------------
# PART C - RNN PARAMETER COUNT
# ------------------------------------------

input_dim = 64
hidden_dim = 128
output_dim = 16

W_xh = input_dim * hidden_dim
W_hh = hidden_dim * hidden_dim
W_hy = hidden_dim * output_dim

b_h = hidden_dim
b_y = output_dim

rnn_total = (
    W_xh
    + W_hh
    + W_hy
    + b_h
    + b_y
)


# ------------------------------------------
# PART D - FC VS CONV PARAMETERS
# ------------------------------------------

# Fully connected layer:
# Input: 128 x 128 x 3
# Neurons: 1024

fc_inputs = 128 * 128 * 3

fc_layer_params = (
    fc_inputs * 1024
    + 1024
)

# Convolutional layer:
# Input channels: 3
# Filters: 64
# Filter size: 3 x 3

conv_layer_params = (
    3 * 3 * 3 * 64
    + 64
)


# ------------------------------------------
# SAVE ARCHITECTURE CSV
# ------------------------------------------

architecture_rows = [
    {
        "layer": "mlp_hidden1",
        "parameters": mlp_hidden1
    },
    {
        "layer": "mlp_hidden2",
        "parameters": mlp_hidden2
    },
    {
        "layer": "mlp_output",
        "parameters": mlp_output
    },
    {
        "layer": "cnn_conv1",
        "parameters": cnn_conv1
    },
    {
        "layer": "cnn_conv2",
        "parameters": cnn_conv2
    },
    {
        "layer": "cnn_dense1",
        "parameters": cnn_dense1
    },
    {
        "layer": "cnn_dense2",
        "parameters": cnn_dense2
    }
]

with open(
    "outputs/part4_architecture.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["layer", "parameters"]
    )

    writer.writeheader()
    writer.writerows(architecture_rows)


# ------------------------------------------
# STORE RESULTS
# ------------------------------------------

results.update({

    # MLP
    "mlp_total": mlp_total,

    # CNN
    "cnn_total": cnn_total,

    # RNN
    "rnn_W_xh": W_xh,
    "rnn_W_hh": W_hh,
    "rnn_W_hy": W_hy,
    "rnn_b_h": b_h,
    "rnn_b_y": b_y,
    "rnn_total": rnn_total,

    # Fully connected vs convolutional
    "fc_layer_params": fc_layer_params,
    "conv_layer_params": conv_layer_params
})


# ------------------------------------------
# SAVE COMPLETE RESULTS FILE
# ------------------------------------------

with open(
    "outputs/part4_results.txt",
    "w",
    encoding="utf-8"
) as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")
        else:
            file.write(f"{key} = {value}\n")


# ------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------

print("\nTask 4.5 - Neural Network Architectures")

print("\nMLP")
print("mlp_hidden1 =", mlp_hidden1)
print("mlp_hidden2 =", mlp_hidden2)
print("mlp_output =", mlp_output)
print("mlp_total =", mlp_total)

print("\nCNN")
print("cnn_conv1 =", cnn_conv1)
print("cnn_conv2 =", cnn_conv2)
print("cnn_dense1 =", cnn_dense1)
print("cnn_dense2 =", cnn_dense2)
print("cnn_total =", cnn_total)

print("\nCNN Output Shapes")
print("input = 28x28x1")
print("conv1 = 28x28x6")
print("pool1 = 14x14x6")
print("conv2 = 10x10x16")
print("pool2 = 5x5x16")
print("flatten = 400")
print("dense1 = 32")
print("dense2 = 10")

print("\nRNN")
print("W_xh =", W_xh)
print("W_hh =", W_hh)
print("W_hy =", W_hy)
print("b_h =", b_h)
print("b_y =", b_y)
print("rnn_total =", rnn_total)

print("\nFully Connected vs Convolutional")
print("fc_layer_params =", fc_layer_params)
print("conv_layer_params =", conv_layer_params)

print("\nSaved: outputs/part4_architecture.csv")
