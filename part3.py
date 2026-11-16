
import csv
import os
import numpy as np

# ==========================================
# TASK 3.1 - K-MEANS FROM SCRATCH
# ==========================================

# 1. Load the complete dataset
with open("data/records.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    data = list(reader)

# 2. Select continuous predictors X1 to X5
predictor_columns = [
    "experience_months",
    "interview_reschedules_90d",
    "salary_gap_ratio",
    "expected_salary_inr_thousands",
    "profile_completeness_score"
]

# 3. Convert predictors into a NumPy matrix
X = np.array([
    [float(row[column]) for column in predictor_columns]
    for row in data
])

# 4. Standardize all five predictors
feature_mean = np.mean(X, axis=0)
feature_sd = np.std(X, axis=0, ddof=1)

X_scaled = (X - feature_mean) / feature_sd

# 5. Implement K-means
def kmeans(X, k):

    n = X.shape[0]

    # Starting centroids using pinned row indices
    step = n // k

    initial_indices = [
        i * step for i in range(k)
    ]

    centroids = X[initial_indices].copy()

    # Initially no rows have assigned clusters
    labels = np.full(n, -1, dtype=int)

    assignment_passes = 0

    while True:

        assignment_passes += 1

        # Calculate squared Euclidean distances
        distances = np.sum(
            (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )

        # np.argmin selects the lower-numbered centroid in a tie
        new_labels = np.argmin(distances, axis=1)

        # Check whether any row changed its cluster
        changed = np.any(new_labels != labels)

        labels = new_labels

        # Update centroids
        new_centroids = centroids.copy()

        for cluster in range(k):

            members = X[labels == cluster]

            if len(members) > 0:
                new_centroids[cluster] = np.mean(
                    members, axis=0
                )

        centroids = new_centroids

        # Stop at the first pass with no assignment changes
        if not changed:
            break

    # Calculate final WCSS
    final_distances = np.sum(
        (X - centroids[labels]) ** 2,
        axis=1
    )

    wcss = np.sum(final_distances)

    return labels, centroids, wcss, assignment_passes


# 6. Run K-means for K = 2, 3, 4, 5, 6
results = {}

cluster_labels = {}

for k in range(2, 7):

    labels, centroids, wcss, passes = kmeans(
        X_scaled, k
    )

    cluster_labels[k] = labels

    results[f"wcss_k{k}"] = wcss

    print(
        f"K = {k}, "
        f"WCSS = {wcss:.4f}, "
        f"assignment_passes = {passes}"
    )


# 7. Save Task 3.1 results
os.makedirs("outputs", exist_ok=True)

with open("outputs/part3_results.txt", "w") as file:

    for key, value in results.items():
        file.write(f"{key} = {value:.4f}\n")


print("\nTask 3.1 Results")

for key, value in results.items():
    print(f"{key} = {value:.4f}")

print("\nSaved: outputs/part3_results.txt")


# ==========================================
# TASK 3.2 - ELBOW METHOD
# ==========================================

elbow_k = None

for k in [3, 4, 5]:

    current_drop = results[f"wcss_k{k}"] - results[f"wcss_k{k+1}"]

    previous_drop = results[f"wcss_k{k-1}"] - results[f"wcss_k{k}"]

    if current_drop < 0.5 * previous_drop:
        elbow_k = k
        break

# Store selected elbow K
results["elbow_k"] = elbow_k

# Rewrite results file
with open("outputs/part3_results.txt", "w") as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")
        else:
            file.write(f"{key} = {value}\n")

print("\nTask 3.2 - Elbow Method")
print("elbow_k =", elbow_k)


# ==========================================
# TASK 3.3 - PROFILE THE THREE SEGMENTS
# ==========================================

# Always use K = 3 for remaining Part 3 tasks
labels_k3 = cluster_labels[3]

# Load outcome and value columns
outcomes = np.array([
    int(row["offer_declined"])
    for row in data
])

values = np.array([
    float(row["days_to_hire"])
    for row in data
])

# Calculate mean X3 for each original cluster
cluster_mean_x3 = {}

for cluster in range(3):

    members = X[labels_k3 == cluster]

    cluster_mean_x3[cluster] = np.mean(
        members[:, 2]
    )

# Sort clusters by ascending mean X3
ordered_clusters = sorted(
    cluster_mean_x3.keys(),
    key=lambda cluster: cluster_mean_x3[cluster]
)

# Map original clusters to A, B, C
segment_names = ["A", "B", "C"]

segment_labels = np.empty(len(labels_k3), dtype="<U1")

for segment, cluster in zip(
    segment_names, ordered_clusters
):

    segment_labels[labels_k3 == cluster] = segment


# Calculate segment profiles
segment_rows = []

for segment, cluster in zip(
    segment_names, ordered_clusters
):

    mask = labels_k3 == cluster

    segment_X = X[mask]
    segment_outcomes = outcomes[mask]
    segment_values = values[mask]

    segment_size = len(segment_X)

    mean_features = np.mean(segment_X, axis=0)

    outcome_rate = np.mean(segment_outcomes)

    mean_value = np.mean(segment_values)

    # Store required segment sizes
    results[f"size_segment_{segment}"] = segment_size

    segment_rows.append({
        "segment": segment,
        "size": segment_size,
        "mean_X1": f"{mean_features[0]:.4f}",
        "mean_X2": f"{mean_features[1]:.4f}",
        "mean_X3": f"{mean_features[2]:.4f}",
        "mean_X4": f"{mean_features[3]:.4f}",
        "mean_X5": f"{mean_features[4]:.4f}",
        "outcome_rate": f"{outcome_rate:.4f}",
        "mean_value": f"{mean_value:.4f}"
    })


# Save segment profiles CSV
with open(
    "outputs/part3_segments.csv",
    "w",
    newline=""
) as file:

    fieldnames = [
        "segment",
        "size",
        "mean_X1",
        "mean_X2",
        "mean_X3",
        "mean_X4",
        "mean_X5",
        "outcome_rate",
        "mean_value"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(segment_rows)


# Rewrite Part 3 results
with open("outputs/part3_results.txt", "w") as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")
        else:
            file.write(f"{key} = {value}\n")


# Display results
print("\nTask 3.3 - Segment Profiles")

for row in segment_rows:
    print(row)

print("\nSegment Sizes")

for segment in segment_names:
    print(
        f"size_segment_{segment} = "
        f"{results[f'size_segment_{segment}']}"
    )

print("\nTotal segment size =", sum(
    results[f"size_segment_{segment}"]
    for segment in segment_names
))

print("\nSaved: outputs/part3_segments.csv")


# ==========================================
# TASK 3.4 - HIERARCHICAL CLUSTERING
# ==========================================

# Pairwise dissimilarity matrix
cohorts = ["E1", "E2", "E3", "E4", "E5", "E6"]

distance_matrix = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
])


# Calculate linkage distance between two clusters
def linkage_distance(cluster_a, cluster_b, method):

    pair_distances = [
        distance_matrix[i, j]
        for i in cluster_a
        for j in cluster_b
    ]

    if method == "single":
        return min(pair_distances)

    elif method == "complete":
        return max(pair_distances)

    elif method == "average":
        return sum(pair_distances) / len(pair_distances)


# Convert cluster indices into names
def cluster_name(cluster):

    return "+".join(
        cohorts[i]
        for i in sorted(cluster)
    )


# Perform agglomerative clustering
def hierarchical_clustering(method):

    # Start with six individual clusters
    clusters = [
        [i] for i in range(len(cohorts))
    ]

    merge_rows = []
    merge_heights = []

    for merge_no in range(1, 6):

        best_distance = float("inf")
        best_i = None
        best_j = None

        # Find the closest pair of clusters
        for i in range(len(clusters)):

            for j in range(i + 1, len(clusters)):

                distance = linkage_distance(
                    clusters[i],
                    clusters[j],
                    method
                )

                if distance < best_distance:
                    best_distance = distance
                    best_i = i
                    best_j = j

        # Select the two clusters
        cluster_a = clusters[best_i]
        cluster_b = clusters[best_j]

        # Ensure cluster_a has the smaller minimum member
        if min(cluster_a) > min(cluster_b):
            cluster_a, cluster_b = cluster_b, cluster_a

        name_a = cluster_name(cluster_a)
        name_b = cluster_name(cluster_b)

        # Record merge
        merge_rows.append({
            "linkage": method,
            "merge_no": merge_no,
            "cluster_a": name_a,
            "cluster_b": name_b,
            "height": f"{best_distance:.4f}"
        })

        merge_heights.append(best_distance)

        # Merge the two clusters
        merged_cluster = sorted(cluster_a + cluster_b)

        # Remove old clusters and add merged cluster
        remaining_clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (best_i, best_j)
        ]

        remaining_clusters.append(merged_cluster)

        # Keep clusters ordered by smallest member
        clusters = sorted(
            remaining_clusters,
            key=lambda cluster: min(cluster)
        )

    return merge_rows, merge_heights


# Run all three linkage methods
all_hierarchical_rows = []

hierarchical_results = {}

for method in ["single", "complete", "average"]:

    rows, heights = hierarchical_clustering(method)

    all_hierarchical_rows.extend(rows)

    # Calculate consecutive merge-height gaps
    gaps = [
        heights[i + 1] - heights[i]
        for i in range(4)
    ]

    # First largest gap wins in case of a tie
    largest_gap_index = int(np.argmax(gaps))

    # Number of merges below the cut
    clusters_at_cut = 6 - (largest_gap_index + 1)

    hierarchical_results[
        f"{method}_clusters_at_cut"
    ] = clusters_at_cut


# Save hierarchical clustering CSV
with open(
    "outputs/part3_hierarchical.csv",
    "w",
    newline=""
) as file:

    fieldnames = [
        "linkage",
        "merge_no",
        "cluster_a",
        "cluster_b",
        "height"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_hierarchical_rows)


# Add cut results to Part 3 results
results.update(hierarchical_results)


# Rewrite results file
with open("outputs/part3_results.txt", "w") as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")
        else:
            file.write(f"{key} = {value}\n")


# Display results
print("\nTask 3.4 - Hierarchical Clustering")

for row in all_hierarchical_rows:
    print(row)

print("\nClusters at cut")

for key, value in hierarchical_results.items():
    print(f"{key} = {value}")

print("\nSaved: outputs/part3_hierarchical.csv")


# ==========================================
# TASK 3.5 - PRINCIPAL COMPONENT ANALYSIS
# ==========================================

# STEP 1: Mean vector of standardized data
pca_mean = np.mean(X_scaled, axis=0)

# STEP 2: Center the standardized data
X_centered = X_scaled - pca_mean

# STEP 3: Calculate covariance matrix
n = X_centered.shape[0]

covariance_matrix = (
    X_centered.T @ X_centered
) / (n - 1)


# STEP 4: Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(
    covariance_matrix
)


# STEP 5: Sort eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]


# Apply sign convention
for j in range(eigenvectors.shape[1]):

    largest_index = np.argmax(
        np.abs(eigenvectors[:, j])
    )

    if eigenvectors[largest_index, j] < 0:
        eigenvectors[:, j] *= -1


# STEP 6: Keep top two eigenvectors
W = eigenvectors[:, :2]


# STEP 7: Project data onto first two components
projections = X_centered @ W


# ==========================================
# PCA RESULTS
# ==========================================

# Sum of all eigenvalues
eigenvalue_sum = np.sum(eigenvalues)

results["eigenvalue_sum"] = eigenvalue_sum

# Individual eigenvalues
for i, value in enumerate(eigenvalues, start=1):
    results[f"eigenvalue_{i}"] = value


# Explained variance ratios
explained_ratios = eigenvalues / eigenvalue_sum

for i, ratio in enumerate(explained_ratios, start=1):
    results[f"explained_ratio_{i}"] = ratio


# Cumulative explained variance ratios
cumulative_ratios = np.cumsum(explained_ratios)

for i, ratio in enumerate(cumulative_ratios, start=1):
    results[f"cumulative_ratio_{i}"] = ratio


# Smallest number of components explaining at least 80%
components_for_80pct = int(
    np.argmax(cumulative_ratios >= 0.80) + 1
)

results["components_for_80pct"] = components_for_80pct


# PC1 and PC2 loadings
for i in range(5):

    results[f"pc1_loading_X{i+1}"] = W[i, 0]

    results[f"pc2_loading_X{i+1}"] = W[i, 1]


# First three data row projections
for row_index in range(3):

    results[f"row{row_index+1}_pc1"] = projections[
        row_index, 0
    ]

    results[f"row{row_index+1}_pc2"] = projections[
        row_index, 1
    ]


# Mean PC1 score for each segment
for segment in ["A", "B", "C"]:

    mask = segment_labels == segment

    mean_pc1 = np.mean(
        projections[mask, 0]
    )

    results[f"mean_pc1_segment_{segment}"] = mean_pc1


# ==========================================
# SAVE COMPLETE PART 3 RESULTS
# ==========================================

with open("outputs/part3_results.txt", "w") as file:

    for key, value in results.items():

        if isinstance(value, (float, np.floating)):
            file.write(f"{key} = {value:.4f}\n")

        else:
            file.write(f"{key} = {value}\n")


# ==========================================
# DISPLAY PCA RESULTS
# ==========================================

print("\nTask 3.5 - PCA Results")

for key, value in results.items():

    if key.startswith((
        "eigenvalue",
        "explained_ratio",
        "cumulative_ratio",
        "components_for",
        "pc1_loading",
        "pc2_loading",
        "row",
        "mean_pc1"
    )):

        if isinstance(value, (float, np.floating)):
            print(f"{key} = {value:.4f}")

        else:
            print(f"{key} = {value}")


print("\nPCA checkpoint:")
print(f"eigenvalue_sum = {eigenvalue_sum:.4f}")

print("\nSaved: outputs/part3_results.txt")
