
import csv
import numpy as np
import os

# ==========================================
# PART 1: ESTIMATION, INFERENCE & REGRESSION
# ==========================================

# Load dataset
with open("data/records.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Extract value column
values = np.array(
    [float(row["days_to_hire"]) for row in data]
)

# ==========================================
# TASK 1.1: ESTIMATION WITH CONFIDENCE INTERVAL
# ==========================================

# 1. Sample size
n = len(values)

# 2. Sample mean
mean_value = np.mean(values)

# 3. Sample standard deviation
sd_value = np.std(values, ddof=1)

# 4. Standard error
se_value = sd_value / np.sqrt(n)

# 5. 95% confidence interval
ci95_low = mean_value - 1.96 * se_value
ci95_high = mean_value + 1.96 * se_value

# Store results
results = {
    "n": n,
    "mean_value": mean_value,
    "sd_value": sd_value,
    "se_value": se_value,
    "ci95_low": ci95_low,
    "ci95_high": ci95_high
}

# Create outputs directory if necessary
os.makedirs("outputs", exist_ok=True)


# ==========================================
# TASK 1.2: HYPOTHESIS TESTING
# ==========================================

# Planning assumption
mu_0 = 8.0

# Significance level
alpha = 0.05

# Critical Z value for a two-tailed test
z_critical = 1.96

# Calculate Z statistic
z_stat = (mean_value - mu_0) / se_value

# Decision
reject_h0 = abs(z_stat) > z_critical

# Store results
results.update({
    "z_stat": z_stat,
    "reject_h0": "yes" if reject_h0 else "no"
})


# ==========================================
# TASK 1.3: REQUIRED SAMPLE SIZE
# ==========================================

# Desired margin of error
E = 0.5

# Confidence level critical value
z_value = 1.96

# Calculate minimum required sample size
n_required = int(np.ceil((z_value * sd_value / E) ** 2))

# Store result
results.update({
    "n_required": n_required
})


# ==========================================
# TASK 1.4: SIMPLE LINEAR REGRESSION
# ==========================================

# Extract X4: Expected salary
x = np.array([
    float(row["expected_salary_inr_thousands"])
    for row in data
])

# Y: Days to hire
y = values

# Calculate means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate sample covariance and variance
cov_xy = np.sum((x - x_mean) * (y - y_mean)) / (n - 1)

var_x = np.sum((x - x_mean) ** 2) / (n - 1)

# Calculate slope
reg_slope = cov_xy / var_x

# Calculate intercept
reg_intercept = y_mean - reg_slope * x_mean

# Predict Y using the regression equation
fitted = reg_intercept + reg_slope * x

# Calculate TSS, ESS and RSS
tss = np.sum((y - y_mean) ** 2)

ess = np.sum((fitted - y_mean) ** 2)

rss = np.sum((y - fitted) ** 2)

# Check decomposition identity
tss_equals_ess_plus_rss = (
    abs(tss - ess - rss) < 1e-6 * tss
)

# Calculate R-squared
r2 = 1 - (rss / tss)

# Calculate RMSE
rmse = np.sqrt(rss / n)

# Calculate MAE
mae = np.mean(np.abs(y - fitted))

# Store results
results.update({
    "reg_slope": reg_slope,
    "reg_intercept": reg_intercept,
    "tss": tss,
    "ess": ess,
    "rss": rss,
    "tss_equals_ess_plus_rss":
        "yes" if tss_equals_ess_plus_rss else "no",
    "r2": r2,
    "rmse": rmse,
    "mae": mae
})


# ==========================================
# TASK 1.5: GRADIENT DESCENT
# ==========================================

# Standardise X4 using sample standard deviation
x_sd = np.std(x, ddof=1)

z = (x - x_mean) / x_sd

# Initial parameters
m = 0.0
c = 0.0

# Learning rate
eta = 0.1

# Function to calculate cost
def calculate_cost(y, z, m, c):
    predictions = m * z + c
    errors = y - predictions
    return np.mean(errors ** 2)

# Initial cost before updates
gd_J0 = calculate_cost(y, z, m, c)

# Store intermediate costs
gd_J1 = None
gd_J10 = None
gd_J100 = None
gd_J500 = None

# Run 500 gradient descent updates
for epoch in range(1, 501):

    # Predictions
    predictions = m * z + c

    # Errors
    errors = y - predictions

    # Calculate gradients
    dm = (-2 / n) * np.sum(z * errors)
    dc = (-2 / n) * np.sum(errors)

    # Update parameters
    m = m - eta * dm
    c = c - eta * dc

    # Store cost after specified updates
    if epoch == 1:
        gd_J1 = calculate_cost(y, z, m, c)

    elif epoch == 10:
        gd_J10 = calculate_cost(y, z, m, c)

    elif epoch == 100:
        gd_J100 = calculate_cost(y, z, m, c)

    elif epoch == 500:
        gd_J500 = calculate_cost(y, z, m, c)

# Convert standardised regression line to original X4 scale
gd_slope = m / x_sd

gd_intercept = c - (m * x_mean / x_sd)

# Difference between gradient descent and closed-form slope
gd_slope_gap = abs(gd_slope - reg_slope)

# Store results
results.update({
    "gd_J0": gd_J0,
    "gd_J1": gd_J1,
    "gd_J10": gd_J10,
    "gd_J100": gd_J100,
    "gd_J500": gd_J500,
    "gd_slope": gd_slope,
    "gd_intercept": gd_intercept,
    "gd_slope_gap": gd_slope_gap
})


# ==========================================
# TASK 1.6: RESIDUAL ANALYSIS
# ==========================================

# ---------- RAW MODEL ----------

# Calculate residuals
residuals_raw = y - fitted

# Stable sorting by fitted values
sorted_indices = np.argsort(fitted, kind="stable")

# Divide into lower and upper halves
lower_indices = sorted_indices[:400]
upper_indices = sorted_indices[400:]

# Sample standard deviations
resid_sd_lower_raw = np.std(
    residuals_raw[lower_indices], ddof=1
)

resid_sd_upper_raw = np.std(
    residuals_raw[upper_indices], ddof=1
)

# Residual spread ratio
resid_sd_ratio_raw = (
    resid_sd_upper_raw / resid_sd_lower_raw
)


# ---------- LOG-LOG REGRESSION ----------

# Transform X and Y using natural logarithm
log_x = np.log(x)
log_y = np.log(y)

# Calculate means
log_x_mean = np.mean(log_x)
log_y_mean = np.mean(log_y)

# Calculate slope
log_cov = np.sum(
    (log_x - log_x_mean) * (log_y - log_y_mean)
) / (n - 1)

log_var = np.sum(
    (log_x - log_x_mean) ** 2
) / (n - 1)

loglog_slope = log_cov / log_var

# Calculate intercept
loglog_intercept = (
    log_y_mean - loglog_slope * log_x_mean
)

# Calculate fitted values
log_fitted = (
    loglog_intercept + loglog_slope * log_x
)

# Residuals
log_residuals = log_y - log_fitted

# Regression metrics
log_tss = np.sum((log_y - log_y_mean) ** 2)

log_rss = np.sum(log_residuals ** 2)

loglog_r2 = 1 - (log_rss / log_tss)

loglog_rmse = np.sqrt(log_rss / n)


# ---------- LOG MODEL RESIDUAL SPREAD ----------

# Stable sorting by log-model fitted values
log_sorted_indices = np.argsort(
    log_fitted, kind="stable"
)

log_lower_indices = log_sorted_indices[:400]
log_upper_indices = log_sorted_indices[400:]

# Sample residual standard deviations
resid_sd_lower_log = np.std(
    log_residuals[log_lower_indices], ddof=1
)

resid_sd_upper_log = np.std(
    log_residuals[log_upper_indices], ddof=1
)

# Ratio
resid_sd_ratio_log = (
    resid_sd_upper_log / resid_sd_lower_log
)


# Store results
results.update({
    "resid_sd_lower_raw": resid_sd_lower_raw,
    "resid_sd_upper_raw": resid_sd_upper_raw,
    "resid_sd_ratio_raw": resid_sd_ratio_raw,

    "loglog_intercept": loglog_intercept,
    "loglog_slope": loglog_slope,
    "loglog_r2": loglog_r2,
    "loglog_rmse": loglog_rmse,

    "resid_sd_lower_log": resid_sd_lower_log,
    "resid_sd_upper_log": resid_sd_upper_log,
    "resid_sd_ratio_log": resid_sd_ratio_log
})


# ==========================================
# TASK 1.7: MULTIPLE LINEAR REGRESSION
# ==========================================

# Extract all predictors
X1 = np.array([
    float(row["experience_months"]) for row in data
])

X2 = np.array([
    float(row["interview_reschedules_90d"]) for row in data
])

X3 = np.array([
    float(row["salary_gap_ratio"]) for row in data
])

X5 = np.array([
    float(row["profile_completeness_score"]) for row in data
])

X6 = np.array([
    float(row["competing_offer_flag"]) for row in data
])

X7 = np.array([
    float(row["referred_flag"]) for row in data
])

# Dependent variable: ln(value)
Y = np.log(y)

# Construct design matrix in the exact required order
X_matrix = np.column_stack((
    np.ones(n),
    np.log(x),
    X1,
    X2,
    X3,
    X5,
    X6,
    X7
))

# Normal equation using linear solve
beta = np.linalg.solve(
    X_matrix.T @ X_matrix,
    X_matrix.T @ Y
)

# Predictions
Y_pred = X_matrix @ beta

# Calculate TSS and RSS
mr_tss = np.sum((Y - np.mean(Y)) ** 2)

mr_rss = np.sum((Y - Y_pred) ** 2)

# R-squared
mr_r2 = 1 - (mr_rss / mr_tss)

# Number of predictors (excluding constant)
p = 7

# Adjusted R-squared
mr_adj_r2 = 1 - (
    (1 - mr_r2) * (n - 1) / (n - p - 1)
)

# Adjusted R-squared for log-log model
loglog_adj_r2 = 1 - (
    (1 - loglog_r2) * (n - 1) / (n - 1 - 1)
)

# Store coefficients and metrics
for i in range(8):
    results[f"mr_beta_{i}"] = beta[i]

results.update({
    "mr_r2": mr_r2,
    "mr_adj_r2": mr_adj_r2,
    "loglog_adj_r2": loglog_adj_r2
})


# Save results

with open("outputs/part1_results.txt", "w") as file:
    for key, value in results.items():
        if isinstance(value, str):
            file.write(f"{key} = {value}\n")
        elif isinstance(value, (int, np.integer)):
            file.write(f"{key} = {value}\n")
        else:
            file.write(f"{key} = {value:.4f}\n")


# Display results
print("\nTASK 1.1 RESULTS")
print("--------------------------")
for key, value in results.items():
    if isinstance(value, str):
        print(f"{key} = {value}")
    elif isinstance(value, (int, np.integer)):
        print(f"{key} = {value}")
    else:
        print(f"{key} = {value}")