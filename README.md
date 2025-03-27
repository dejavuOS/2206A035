New Features Guide
1. Loss Function Selection
You can now select different loss functions for both regression and classification algorithms:
For Regression Models:

MSE (Mean Squared Error): Standard loss function that penalizes larger errors more heavily
MAE (Mean Absolute Error): Less sensitive to outliers than MSE
Huber Loss: Combines benefits of MSE and MAE, particularly useful for robust regression

To use:

Select a regression algorithm (Linear Regression or SVR)
Choose your preferred loss function from the dropdown menu
Configure other parameters as needed
Click "Train" to see the results

For Classification Models:

Cross-Entropy: Standard loss function for classification problems
Hinge Loss: Used in support vector machines, focuses on margin maximization

Note: For Logistic Regression with Hinge Loss, the system automatically switches to LinearSVC for proper implementation.
2. Support Vector Regression (SVR)
A powerful regression technique that's now fully supported:

Find SVR under the "Regression" section
Configure parameters:

C: Controls regularization (higher values = less regularization)
epsilon: Controls the width of the insensitive loss region
kernel: Choose between linear, rbf, and poly



SVR works particularly well with the California Housing dataset (which replaced the deprecated Boston Housing dataset).
3. Bayesian Methods
A new Bayesian Methods section has been added with Gaussian Naive Bayes implementation:

Find the "Bayesian Methods" section under the Classical ML tab
Configure parameters:

Var Smoothing: Stability parameter that adds a small portion of the largest variance to all features
Prior Probabilities: Choose between uniform (default) or custom priors
Show Posterior Probabilities: Visualize class probability distributions



For custom priors:

Select "Custom" radio button
Enter comma-separated values that sum to 1.0 (e.g., "0.3,0.7" for binary classification)
Train the model to apply these custom priors

The posterior probability visualization shows the distribution of class probabilities assigned by the model, helping you understand classification confidence.
4. Missing Value Handling
Enhanced data preprocessing with multiple missing value handling strategies:

Find "Missing Value Handling" dropdown in the Data Management section
Choose from:

No Imputation: Skip missing value handling
Mean Imputation: Replace missing values with column means
Median Imputation: Replace missing values with column medians
Most Frequent Imputation: Replace with most common value
Forward Fill: Propagate last valid value forward
Backward Fill: Propagate next valid value backward
Linear Interpolation: Estimate missing values through linear interpolation



Missing value handling is applied before train/test splitting for consistent preprocessing.
5. Neural Network Enhancements
Deep learning tab now supports loss function selection:

Build your neural network architecture by adding layers
Select a loss function:

Cross-Entropy: Standard for classification tasks
MSE: Standard for regression tasks
MAE: Alternative for regression, less sensitive to outliers
Hinge: Support vector approach to classification



The system will automatically configure the final layer and loss function appropriately.
Data Visualization
Visualizations have been enhanced to provide more information:

Regression Models: Shows predicted vs. actual values with a diagnostic line
Classification Models:

2D feature visualization for datasets with 2 features
PCA-reduced 2D visualization for higher-dimensional data


Bayesian Models: Posterior probability distribution visualization
Neural Networks: Training and validation metrics over epochs

Metrics Display
The metrics panel now shows more comprehensive performance statistics:
For Regression:

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
R² Score

For Classification:

Accuracy
F1 Score (weighted)
Confusion Matrix
ROC AUC (for binary classification)

Example Workflow

Load a dataset (e.g., Breast Cancer for classification, California Housing for regression)
Select preprocessing options (scaling method, test split percentage, missing value handling)
Navigate to the appropriate algorithm tab
Configure algorithm parameters and select a loss function
Train the model and analyze the results in the visualization area
Experiment with different parameters to improve performance