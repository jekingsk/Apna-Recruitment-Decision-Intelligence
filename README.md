Domain: Recruitment

## How to run

### Requirements

- Python 3.10 or newer (Python 3.12 is recommended with NumPy 1.26.4)
- NumPy 1.26.4

Install the pinned dependency from the requirements file:

```bash
pip install -r requirements.txt
```

### Project structure

Make sure your project folder contains the following files:

```text
Apna-Recruitment-Decision-Intelligence/
│
├── data/
│   └── records.csv
│
├── outputs/
│
├── requirements.txt
├── generate_data.py
├── ml_helpers.py
├── part1.py
├── part2.py
├── part3.py
├── part4.py
└── README.md
```

### Running the project

If the dataset has not been generated yet, run:

```bash
python generate_data.py
```

After that, execute each part separately:

```bash
python part1.py
python part2.py
python part3.py
python part4.py
```

The scripts automatically generate their respective result files inside the `outputs/` folder.

The project uses Python's standard library and NumPy. The machine-learning algorithms and evaluation metrics have been implemented from scratch rather than using ready-made machine-learning libraries.

---

## Part 1 findings

### 1. Understanding the dataset

The first step was to understand the dataset through basic statistical calculations. It contains 800 observations, and `days_to_hire` was used as the response variable because the supplied CSV does not contain a separate `value` column.

The calculated mean is 8.9382 days, with a standard deviation of 9.1723 days.

The main results are:

- Total observations: 800
- Mean: 8.9382
- Standard deviation: 9.1723
- Standard error: 0.3243
- 95% confidence interval: 8.3026 to 9.5738

The confidence interval gives us an estimated range for the population mean based on the available sample.

### 2. Hypothesis testing

A Z-test was performed to check whether the observed mean provides enough evidence against the null hypothesis.

The calculated Z-statistic was 2.8930. At a 5% significance level, the critical values are -1.96 and +1.96.

Since 2.8930 is greater than 1.96, the null hypothesis was rejected. This means that the result is statistically significant at the selected significance level.

In this recruitment context, a Type I error would mean concluding that the true average time to an offer decision differs from 8 days when it is actually 8 days. With a 5% significance level, the test allows a 5% chance of this error under the assumptions of the test. The estimated mean is 0.9382 days above the planning assumption. That is statistically significant, but it is less than one day; whether it is operationally important depends on the team’s deadlines and service expectations.

### 3. Sample size estimation

The estimated sample size required for the analysis was 1,293 observations. However, the available dataset contains only 800 observations.

This means that another 493 observations would be needed to reach the estimated sample size.

### 4. Simple linear regression

Simple linear regression was used to understand the relationship between X4 (expected monthly salary, in thousands of INR) and the response variable, days to the offer decision.

The resulting regression equation is:

\[
\hat{y}=-3.6598+0.6701x
\]

The slope of 0.6701 suggests that a one-unit increase in the predictor is associated with an estimated increase of 0.6701 units in the response.

The model produced the following results:

| Metric | Value |
|---|---:|
| TSS | 67,221.4285 |
| ESS | 34,925.6088 |
| RSS | 32,295.8197 |
| R² | 0.5196 |
| RMSE | 6.3537 |
| MAE | 3.7636 |

The relationship TSS = ESS + RSS was also verified.

The R² value of 0.5196 means that the fitted model explains approximately 51.96% of the variation in the response variable.

### 5. Gradient descent

After calculating the regression parameters analytically, gradient descent was implemented to estimate the same parameters through repeated updates.

The cost decreased as the number of iterations increased:

| Iteration | Cost |
|---|---:|
| 0 | 163.9178 |
| 1 | 119.4579 |
| 10 | 41.7973 |
| 100 | 40.3698 |
| 500 | 40.3698 |

The cost dropped significantly during the initial iterations and then became stable.

The final slope obtained through gradient descent was 0.6701, matching the analytical regression slope. The difference between the two slopes was 0.0000.

This confirms that gradient descent successfully converged to the regression solution for this dataset.

### 6. Residual analysis and logarithmic transformation

Residual analysis was performed to check whether the model had a consistent level of prediction error across different ranges of the data.

For the raw model, the residual standard deviations were:

- Lower group: 2.3155
- Upper group: 8.6891
- Ratio: 3.7525

The large difference between these values suggests that the residual variance is not constant. This is a sign of heteroscedasticity.

A logarithmic transformation was then applied. After the transformation, the residual standard deviations became:

- Lower group: 0.5165
- Upper group: 0.5369
- Ratio: 1.0396

The two groups now have much more similar residual spreads, suggesting that the transformation improved variance stability.

The log-log regression produced:

| Metric | Value |
|---|---:|
| Intercept | -2.4370 |
| Slope | 1.5030 |
| R² | 0.6565 |
| RMSE | 0.5268 |
| Adjusted R² | 0.6561 |

### 7. Multiple linear regression

Multiple linear regression was used to study how all seven predictors work together in explaining the response variable.

The estimated coefficients were:

| Coefficient | Value |
|---|---:|
| β₀ | -1.8613 |
| β₁ | 1.1576 |
| β₂ | -0.0074 |
| β₃ | 0.0940 |
| β₄ | 0.1067 |
| β₅ | 0.0028 |
| β₆ | 0.1518 |
| β₇ | 0.0172 |

The model achieved an R² of 0.7425 and an adjusted R² of 0.7403.

Compared with the log-log model's adjusted R² of 0.6561, the multiple regression model explains more variation in the response, even after accounting for the number of predictors.

Using the assignment’s absolute-coefficient cut-off of 0.05, the predictors above the cut-off are X4 (log expected salary, β₁ = 1.1576), X2 (interview reschedules, β₃ = 0.0940), X3 (salary gap ratio, β₄ = 0.1067), and X6 (competing-offer flag, β₆ = 0.1518). The coefficients use different units, so this cut-off is only the requested mechanical rule and should not be treated as a ranking of importance. The coefficients show the estimated relationship between each predictor and the response while keeping the other predictors constant.

---

## Part 2 findings

### 1. Preparing the training and testing data

Before building the classification models, the dataset was divided into training and testing sets using a stratified 70:30 split.

The split maintained the proportion of positive and negative outcomes in both sets.

| Dataset | Total records | Positive outcomes |
|---|---:|---:|
| Training | 560 | 119 |
| Testing | 240 | 51 |

The seven predictors were standardized using the training data's mean and sample standard deviation. These same values were then used to transform the test data.

This approach prevents information from the test set from influencing the training process.

### 2. Logistic regression

Logistic regression was implemented from scratch using sigmoid activation, binary cross-entropy loss, and batch gradient descent.

The model was trained with a learning rate of 0.1 for 1,000 iterations.

The results were:

- Initial training loss: 0.6931
- Final training loss: 0.3704
- Test loss: 0.3818
- Bias: -2.0696

The training loss decreased considerably, and the test loss remained reasonably close to it.

The learned weights were:

| Predictor | Weight |
|---|---:|
| X1 | -0.5554 |
| X2 | 0.5505 |
| X3 | 0.9013 |
| X4 | -0.1459 |
| X5 | -0.2162 |
| X6 | 0.2348 |
| X7 | -0.0120 |

These weights help explain how changes in the predictors influence the model's estimated probability of an offer being declined.

### 3. Comparing learning rates

To understand how the learning rate affects training, the three values required by the assignment were tested. Each model was restarted with zero weights and bias and trained for exactly 200 updates.

| Learning rate | Final training loss after 200 updates |
|---:|---:|
| 1.0 | 0.3703 |
| 0.1 | 0.3776 |
| 0.01 | 0.5091 |

From lowest to highest, the losses are 0.3703, 0.3776, and 0.5091. A learning rate that is too small makes learning slow, while a very large rate can overshoot the minimum; in these tests, 1.0 gave the lowest loss after 200 updates. The main logistic regression model was kept at the required learning rate of 0.1 for the remaining experiments.

### 4. Understanding the odds ratios

The odds ratios were calculated by taking the exponential of each learned logistic regression coefficient.

| Predictor | Odds ratio |
|---|---:|
| X1 | 0.5739 |
| X2 | 1.7341 |
| X3 | 2.4629 |
| X4 | 0.8642 |
| X5 | 0.8056 |
| X6 | 1.2646 |
| X7 | 0.9881 |

An odds ratio greater than 1 means that an increase in the corresponding standardized predictor is associated with higher odds of the positive outcome. A value below 1 indicates lower odds, assuming the other predictors remain unchanged.

The three largest absolute model weights are X3 (0.9013), X1 (0.5554), and X2 (0.5505). X3 has the largest odds ratio at 2.4629, meaning that a one-standard-deviation increase in salary-gap ratio multiplies the estimated odds of an offer being declined by about 2.46, holding the other predictors constant.

### 5. Evaluating different classification thresholds

The logistic regression model was tested using two thresholds: 0.5 and 0.3.

| Metric | Threshold 0.5 | Threshold 0.3 |
|---|---:|---:|
| TP | 24 | 38 |
| FP | 20 | 44 |
| TN | 169 | 145 |
| FN | 27 | 13 |
| Accuracy | 0.8042 | 0.7625 |
| Precision | 0.5455 | 0.4634 |
| Recall | 0.4706 | 0.7451 |
| F1-score | 0.5053 | 0.5714 |
| Specificity | 0.8942 | 0.7672 |

The difference between the two thresholds is quite noticeable.

At 0.3, the model identifies more candidates who eventually decline an offer. Recall increases from 0.4706 to 0.7451, but the number of false alarms also increases.

At 0.5, the model is more conservative. It produces fewer false positives but misses more positive cases. Lowering the threshold from 0.5 to 0.3 raises recall from 0.4706 to 0.7451, while precision falls from 0.5455 to 0.4634.

The F1-score is higher at 0.3, while accuracy and specificity are higher at 0.5.

A simple model that predicts every candidate as negative achieves an accuracy of 0.7875. This is a useful reminder that accuracy alone does not tell us how well a model handles an imbalanced dataset.

### 6. Threshold sweep and ROC analysis

The model was also tested across nine thresholds, from 0.1 to 0.9, to observe how the true positive rate and false positive rate change.

| Threshold | TPR | FPR |
|---|---:|---:|
| 0.1 | 0.9412 | 0.4656 |
| 0.2 | 0.8431 | 0.3386 |
| 0.3 | 0.7451 | 0.2328 |
| 0.4 | 0.6078 | 0.1746 |
| 0.5 | 0.4706 | 0.1058 |
| 0.6 | 0.3333 | 0.0476 |
| 0.7 | 0.1373 | 0.0212 |
| 0.8 | 0.0392 | 0.0000 |
| 0.9 | 0.0000 | 0.0000 |

The results show the expected trade-off. Lowering the threshold helps identify more positive cases, but it also increases the number of false alarms. Increasing the threshold reduces false positives but causes the model to miss more candidates.

### 7. K-nearest neighbours

KNN was implemented without using a ready-made machine-learning library. Euclidean distance was used to find the nearest observations, and five-fold cross-validation was performed to select the best value of K.

The candidate values were odd numbers from 1 to 21.

| K | Mean CV F1-score |
|---|---:|
| 1 | 0.4249 |
| 3 | 0.4249 |
| 5 | 0.4541 |
| 7 | 0.4165 |
| 9 | 0.4212 |
| 11 | 0.4382 |
| 13 | 0.3993 |
| 15 | 0.3965 |
| 17 | 0.3655 |
| 19 | 0.3736 |
| 21 | 0.3853 |

K = 5 gave the highest mean cross-validation F1-score of 0.4541, so it was selected for the final evaluation.

Its test-set performance was:

| Metric | Value |
|---|---:|
| TP | 17 |
| FP | 24 |
| TN | 165 |
| FN | 34 |
| Accuracy | 0.7583 |
| Precision | 0.4146 |
| Recall | 0.3333 |
| F1-score | 0.3696 |

On this test set, logistic regression achieved F1 = 0.5714 and recall = 0.7451, compared with KNN’s F1 = 0.3696 and recall = 0.3333. Logistic regression therefore performed better on both measures.

### 8. Choosing a threshold based on financial cost

Classification performance is important, but in an actual recruitment process, mistakes also have a financial impact.

For this experiment, the following costs were considered:

- A false alarm costs INR 50 because it leads to an unnecessary recruiter follow-up.
- A missed positive costs INR 400 because the company may need to reopen the search after an offer is declined.

The total cost is calculated as:

\[
Cost=50(FP)+400(FN)
\]

Instead of automatically choosing 0.3 or 0.5, thresholds from 0.05 to 0.95 were evaluated using training-set probabilities. The threshold with the lowest training cost was selected.

The selected threshold was 0.15.

| Evaluation | Threshold or scenario | Cost |
|---|---|---:|
| Training | 0.15 | INR 12,150 |
| Testing | 0.15 | INR 6,050 |
| Testing | 0.50 | INR 11,800 |
| Testing | Flag nobody | INR 20,400 |

The results show why choosing a threshold based only on accuracy or F1-score may not always be the best business decision.

At the selected threshold of 0.15, the test cost is lower than both the cost at 0.5 and the cost of not flagging any candidates.

I would deploy logistic regression as a recruiter decision-support model using the cost-selected threshold of 0.15. The test cost at this threshold was INR 6,050, compared with INR 11,800 at threshold 0.50 and INR 20,400 when nobody is flagged. The threshold was selected using training data to avoid tuning it to the test set; the test set was kept for final evaluation.

---

## Part 3 findings

### 1. K-means clustering

K-means clustering was used to divide candidates into groups based on similarities in their recruitment-related features.

The first five predictors were selected:

- X1: Experience months
- X2: Interview reschedules
- X3: Salary gap ratio
- X4: Expected salary
- X5: Profile completeness

Before clustering, these features were standardized so that variables with larger numerical values would not dominate the distance calculations.

The WCSS values for different cluster counts were:

| K | WCSS |
|---|---:|
| 2 | 2262.6753 |
| 3 | 1602.8471 |
| 4 | 1494.7732 |
| 5 | 1323.7157 |
| 6 | 1278.1691 |

Based on the elbow method, K = 3 was selected.

The successive WCSS drops are 659.8282 (K = 2 to 3), 108.0739 (K = 3 to 4), 171.0575 (K = 4 to 5), and 45.5466 (K = 5 to 6). The elbow rule supports K = 3 because the drop from 3 to 4 is less than half the drop from 2 to 3. The drops do not shrink steadily: the pinned starting rows give a reproducible result, but not necessarily the global optimum for every K.

### 2. Understanding the recruitment segments

After selecting three clusters, their average feature values and outcome rates were examined to understand what makes each group different.

| Segment | Size | Mean X1 | Mean X2 | Mean X3 | Mean X4 | Mean X5 | Outcome rate | Mean value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 322 | 47.5028 | 1.9503 | 0.2250 | 12.5301 | 62.1584 | 0.0186 | 3.5976 |
| B | 255 | 28.6071 | 6.0706 | 0.4534 | 28.7192 | 55.0082 | 0.2353 | 17.1248 |
| C | 223 | 14.3892 | 4.2287 | 0.7118 | 16.5099 | 38.5215 | 0.4664 | 7.2883 |

The `mean_value` column uses `days_to_hire`, since the actual dataset does not contain a separate `value` column.

Based on these results, the three groups can be described as follows.

**Segment A — Experienced, Low-Friction Pipeline**

This group has the highest average experience and profile completeness. It also has the fewest interview reschedules and a very low outcome rate of 1.86%.

These candidates appear to move through the recruitment process with relatively few difficulties.

A suitable approach would be to keep the interview process smooth and avoid unnecessary delays when making offers.

**Segment B — High-Rescheduling Pipeline**

This group has the highest average number of interview reschedules and the highest expected salary among the three segments. Its outcome rate is 23.53%.

Frequent scheduling changes may make the recruitment process difficult to manage, while compensation expectations may require additional discussion.

Recruiters could focus on better scheduling coordination, regular follow-ups, and discussing salary expectations earlier.

**Segment C — High Salary-Gap Pipeline**

This group has the lowest average experience and profile completeness, along with the highest salary-gap ratio. It also has the highest outcome rate at 46.64%.

This suggests that compensation expectations and incomplete candidate profiles may deserve additional attention in this segment.

Recruiters could discuss compensation early and help candidates complete missing profile information. Segment C has the highest outcome rate, at 0.4664 (46.64%).

These segments give the recruitment team a clearer picture of different candidate groups and can help them plan more suitable engagement strategies.

### 3. Hierarchical clustering

Hierarchical clustering was performed using single, complete, and average linkage.

The purpose was to observe how individual observations combine into larger clusters as the distance between them increases.

| Linkage | Merge sequence | Final merge height |
|---|---|---:|
| Single | E3+E6, then E2, then E4+E5, then E1 | 3.4000 |
| Complete | E3+E6, then E2, then E4+E5, then E1 | 7.9000 |
| Average | E3+E6, then E2, then E4+E5, then E1 | 5.6556 |

At the specified cut, single linkage produces four clusters: {E1}, {E2}, {E3, E6}, and {E4, E5}. Complete linkage produces two clusters: {E1, E4, E5} and {E2, E3, E6}. Average linkage also produces two clusters.

Single linkage uses the closest pair of points to merge groups, so it can create a chain of connected clusters. Complete linkage uses the farthest pair, so it keeps groups separate until all their members are relatively close; that is why its cut gives two larger clusters here.

This explains why the final merge heights and cluster counts differ across the methods.

### 4. Principal Component Analysis

PCA was used to reduce the number of dimensions while retaining as much information from the original five features as possible.

The standardised feature matrix has a mean vector of five zeros (apart from tiny floating-point rounding), so the data was centred before calculating the covariance matrix and carrying out eigenvalue decomposition.

The explained variance results were:

| Component | Eigenvalue | Explained variance | Cumulative variance |
|---|---:|---:|---:|
| PC1 | 2.6136 | 52.27% | 52.27% |
| PC2 | 1.1828 | 23.66% | 75.93% |
| PC3 | 0.5114 | 10.23% | 86.16% |
| PC4 | 0.4055 | 8.11% | 94.27% |
| PC5 | 0.2867 | 5.73% | 100.00% |

The first two components explain 75.93% of the total variance. Three components are enough to retain more than 80%, capturing 86.16% of the variance.

The loadings for the first two components were:

| Feature | PC1 | PC2 |
|---|---:|---:|
| X1 | 0.5317 | 0.1773 |
| X2 | -0.4100 | 0.5056 |
| X3 | -0.5187 | -0.2648 |
| X4 | -0.2916 | 0.6934 |
| X5 | 0.4418 | 0.4025 |

PC1 has strong positive loadings for experience and profile completeness. Salary gap ratio and interview reschedules have negative loadings.

The average PC1 scores for the three segments are:

- Segment A: 1.7773
- Segment B: -0.7831
- Segment C: -1.6709

These differences show that the principal components also help distinguish the characteristics of the three recruitment segments.

---

## Part 4 findings

### 1. Selecting the best decision tree split

Four possible splits were tested on the training data using information gain and Gini gain.

The initial impurity values were:

- Root entropy: 0.7462
- Root Gini: 0.3347

The results were:

| Split | Condition | Information gain | Gini gain |
|---|---|---:|---:|
| S1 | X6 = 1 | 0.0110 | 0.0052 |
| S2 | X7 = 1 | 0.0033 | 0.0015 |
| S3 | X3 > 0.5 | 0.1046 | 0.0494 |
| S4 | X1 ≤ 24 | 0.1252 | 0.0606 |

S4 produced the highest information gain and Gini gain, so it was selected as the best split.

In simple terms, dividing candidates according to whether their experience is 24 months or less produced the greatest reduction in impurity among the four available choices.

A decision tree that keeps splitting until every leaf is pure can easily overfit the training data. It may start learning individual records instead of general patterns.

One way to avoid this is to set a maximum tree depth or require a minimum number of observations in each leaf.

### 2. Decision stump and voting models

The selected decision stump uses the rule:

\[
X1 \leq 24
\]

The positive outcome rates for the two branches were:

- Yes branch: 0.4574
- No branch: 0.0887

The stump and the other models were then compared on the same test set. A threshold of 0.30 was used for their predictions.

| Model | TP | FP | TN | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 38 | 44 | 145 | 13 | 0.4634 | 0.7451 | 0.5714 |
| KNN | 17 | 24 | 165 | 34 | 0.4146 | 0.3333 | 0.3696 |
| Decision stump | 39 | 51 | 138 | 12 | 0.4333 | 0.7647 | 0.5532 |
| Hard voting | 35 | 45 | 144 | 16 | 0.4375 | 0.6863 | 0.5344 |
| Soft voting | 38 | 49 | 140 | 13 | 0.4368 | 0.7451 | 0.5507 |

Logistic regression achieved the highest F1-score of 0.5714. The decision stump had the highest recall at 0.7647.

Both voting methods performed reasonably well, but neither improved upon the logistic regression F1-score.

I would ship logistic regression: it has the highest F1-score (0.5714) with recall of 0.7451. Hard voting reached F1 = 0.5344 and recall = 0.6863, while soft voting reached F1 = 0.5507 and recall = 0.7451. For this dataset, combining the three models did not improve on logistic regression’s F1-score.

### 3. AdaBoost

AdaBoost was implemented from scratch using eight observations and two weak classifiers.

All observations started with an equal weight of 1/8. After the first classifier made its predictions, the weights were updated so that incorrectly classified observations received greater importance.

The results were:

| Metric | Result |
|---|---:|
| Round 1 error | 0.2500 |
| Round 1 alpha | 0.5493 |
| Round 2 error | 0.1667 |
| Round 2 alpha | 0.8047 |
| Misclassified weight after Round 1 | 0.5000 |
| Ensemble accuracy | 0.7500 |

After the first round, the two incorrectly classified observations together held 50% of the total normalized weight.

The second classifier received a higher alpha because its weighted error was lower than the first classifier's error. This gives the second classifier more influence in the final ensemble.

### 4. Out-of-bag probability

The out-of-bag probability was calculated for a bootstrap sample containing 560 draws from 560 training observations, with replacement.

The probability that a particular observation is never selected is:

\[
P(OOB)=\left(1-\frac{1}{560}\right)^{560}
\]

The resulting OOB fraction is 0.3676.

This means that approximately 36.76% of the training observations are expected to be left out of an individual bootstrap sample. Those observations can be used for out-of-bag evaluation.

### 5. Neural network architecture calculations

This task involved calculating the number of trainable parameters in different neural network architectures. The calculations were done manually using the specified layer dimensions.

#### Model A — Multi-Layer Perceptron (MLP)

The MLP architecture is:

\[
784 \rightarrow 16 \rightarrow 16 \rightarrow 10
\]

The parameter counts are:

| Layer | Parameters |
|---|---:|
| Hidden layer 1 | 12,560 |
| Hidden layer 2 | 272 |
| Output layer | 170 |
| **Total** | **13,002** |

#### Model A — Convolutional Neural Network (CNN)

The CNN takes a 28 × 28 grayscale image as input. It uses two convolutional layers, followed by max-pooling, flattening, and two dense layers.

| Layer | Output shape | Parameters |
|---|---|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv1 | 28 × 28 × 6 | 156 |
| MaxPool1 | 14 × 14 × 6 | 0 |
| Conv2 | 10 × 10 × 16 | 2,416 |
| MaxPool2 | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense1 | 32 | 12,832 |
| Dense2 | 10 | 330 |
| **Total** | | **15,734** |

The convolution output dimensions were calculated using:

\[
Output=\frac{Input-Filter+2(Padding)}{Stride}+1
\]

ReLU is used in the hidden layers. For the ten-class output, softmax is suitable because it converts the output scores into class probabilities. Categorical cross-entropy is an appropriate loss function for this classification setup.

#### Model B — Recurrent Neural Network (RNN)

The RNN uses an input dimension of 64, a hidden dimension of 128, and an output dimension of 16.

| Parameter | Shape | Parameters |
|---|---|---:|
| W_xh | 64 × 128 | 8,192 |
| W_hh | 128 × 128 | 16,384 |
| W_hy | 128 × 16 | 2,048 |
| b_h | 128 | 128 |
| b_y | 16 | 16 |
| **Total** | | **26,768** |

One important property of an RNN is that it reuses the same weight matrices at every time step. Because of this, the number of trainable parameters stays the same even when the input sequence becomes longer.

#### Fully connected layer versus convolutional layer

The final comparison used an input of 128 × 128 × 3, a fully connected layer with 1,024 neurons, and a convolutional layer with 64 filters of size 3 × 3.

| Layer type | Parameters |
|---|---:|
| Fully connected | 50,332,672 |
| Convolutional | 1,792 |

The difference is very large. A fully connected layer connects every input value to every neuron, while a convolutional layer reuses the same filters across different parts of the image.

This is why convolutional layers are generally much more parameter-efficient for image-related tasks.

In deep networks, vanishing gradients can make it difficult for early layers to learn because gradients become very small during backpropagation. Skip connections help address this problem by providing shorter paths through which information and gradients can travel.

---

## Model-choice memo

**To:** Head of Talent Acquisition Operations  
**Subject:** Recommendation for Production Recruitment Decision Intelligence

I recommend using **Logistic Regression** as the primary model for identifying candidates who may decline an offer.

On the test set, logistic regression achieved an F1-score of **0.5714** and recall of **0.7451** at a threshold of 0.30. Its test loss was 0.3818. The decision stump achieved slightly higher recall (0.7647), but its F1-score was lower at 0.5532. KNN achieved an F1-score of 0.3696. Neither hard voting (0.5344) nor soft voting (0.5507) performed better than logistic regression.

For production, I recommend using a financially selected threshold of **0.15**, chosen using training data. The resulting training cost was INR 12,150, while the test cost was INR 6,050. This is lower than the test cost of INR 11,800 at threshold 0.50 and INR 20,400 when no candidates are flagged.

One important risk is that false alarms may lead to unnecessary recruiter follow-ups. The model should therefore be used to support recruiters rather than make automatic rejection decisions. Its performance and threshold should also be reviewed as recruitment patterns change.

I would not prioritize building Model A (MLP/CNN) or Model B (RNN) before deploying the structured-data model. Logistic regression is a simpler and more suitable starting point for the current dataset. More complex architectures can be explored later if relevant image, document, or sequential data becomes available.

**Recommendation:** Start with logistic regression at threshold 0.15, keep recruiters involved in the final decisions, and monitor model performance regularly.

---

## Limitations

There are a few limitations to keep in mind when interpreting the results of this project.

1. **Synthetic data:** The experiments use synthetic recruitment data. Real recruitment decisions can be influenced by many factors that may not be represented in this dataset.

2. **Dataset size:** The dataset contains 800 observations. A larger dataset would help us understand whether the same patterns hold across different companies, roles, and candidate groups.

3. **Class imbalance:** Positive outcomes are less common than negative outcomes. Because of this, accuracy alone is not enough to judge model performance. Metrics such as precision, recall, and F1-score are also important.

4. **Limited predictors:** The models use seven structured features. Factors such as communication, job-market conditions, company reputation, and candidate preferences are not included.

5. **Threshold and cost assumptions:** The selected threshold depends on the assumed costs of false alarms and missed positive cases. These costs may be different for different organizations, so the threshold should be reviewed when business conditions change.

6. **No live deployment testing:** The models have been evaluated using a held-out test set, but they have not yet been tested in an actual recruitment workflow.

7. **Cluster interpretation:** The clusters are formed using numerical similarities. Their labels and suggested recruitment strategies are interpretations of the results, not proof that a particular candidate will behave in a specific way.

8. **Fairness considerations:** Before using the model in practice, it should be checked for unintended bias. Predictions should not result in unfair treatment of candidates.

9. **Neural network evaluation:** The MLP, CNN, and RNN architectures were analyzed mathematically, but they were not trained and tested on the recruitment dataset. Their actual performance on this problem is therefore still unknown.