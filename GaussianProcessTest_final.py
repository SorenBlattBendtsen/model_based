#%%
import pandas as pd
import numpy as np
from prepare_country_data import transpose_for_country_code, split_and_normalize, cyclical_transformation
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("nordic_energy_data.csv")
df_dk1 = transpose_for_country_code(df, "DK_1")
df_dk1 = cyclical_transformation(df_dk1)


X_train_df, X_test_df, y_train_df, y_test_df, x_train_mean, x_train_std, y_train_mean, y_train_std = split_and_normalize(df_dk1)
N_train = len(X_train_df)

# amount of data
data_len = 500

X_train_df = X_train_df.iloc[-data_len:,:]
y_train_df = y_train_df.iloc[-data_len:]
nan_rows = X_train_df[X_train_df.isnull().any(axis=1)]

y_test_df = y_test_df.iloc[:data_len]
X_test_df = X_test_df.iloc[:data_len]




# To numpy arrays
X_train = X_train_df.to_numpy()
X_test = X_test_df.to_numpy()
y_train = y_train_df.to_numpy().ravel()  # ravel() to convert y_train to 1D array
y_test = y_test_df.to_numpy().ravel()  # ravel() to convert y_test to 1D array


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#%%

# Define the kernel
ESS_c = 1.0
ESS_ls = 5.0
ESS_per = 24
WK_nl = 1
rbf_c = 1
rbf_ls = 1
# RBF = SE kernel

kernel = ESS_c * ExpSineSquared(length_scale=ESS_ls, periodicity=ESS_per) + WhiteKernel(noise_level=WK_nl) + rbf_c * RBF(length_scale=rbf_ls)
#kernel = ESS_c * ExpSineSquared(length_scale=ESS_ls, periodicity=ESS_per) + WhiteKernel(noise_level=WK_nl)
#kernel = rbf_c * RBF(length_scale=rbf_ls)
#kernel = ESS_c * ExpSineSquared(length_scale=ESS_ls, periodicity=ESS_per) + WhiteKernel(noise_level=WK_nl)
#kernel = rbf_c * RBF(length_scale=rbf_ls) + WhiteKernel(noise_level=WK_nl)
#kernel = rbf_c * RBF(length_scale=rbf_ls) + ESS_c * ExpSineSquared(length_scale=ESS_ls, periodicity=ESS_per)
#kernel = rbf_c * RBF(length_scale=rbf_ls)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit the Gaussian Process model
gp.fit(X_train, y_train)

# Make predictions on the test set
y_pred, y_std = gp.predict(X_test, return_std=True)

# Plotting the results
plt.figure(figsize=(10, 5))
#plt.errorbar(np.arange(len(y_test)), y_test, yerr=y_std, label='True values with error bars', fmt='o', color='r')
plt.plot(np.arange(len(y_test)), y_test, 'r', label='True values', linewidth=2)
plt.plot(np.arange(len(y_test)), y_pred, 'k', label='Predictions', linewidth=2)
plt.title('Gaussian Process Predictions with Periodic Kernel')
plt.xlabel('Data Point')
plt.ylabel('Day-ahead Electricity Price')
plt.legend()
plt.show()

# Plotting the covariance matrix
K = kernel(X_train)  # Compute the covariance matrix for the training data

# Plotting the covariance matrix
plt.figure(figsize=(10, 8))
plt.imshow(K, interpolation='none', cmap='viridis')
plt.colorbar(label='Covariance')
plt.title('Covariance Matrix of the Gaussian Process')
plt.xlabel('Data Index')
plt.ylabel('Data Index')
plt.show()

# It may show only covariance in the middle when we sum many kernels.
# Also, we saw none of the kernels worked well alone. We need to combine the white noise with one of the others.

#%%

# To provide insights into feature importance in a Gaussian Process (GP) model, 
# we can use a method that examines the sensitivity of the model's predictions to
# perturbations in each feature. This method can be understood as a local approximation
# of feature importance, and it involves observing how the predictions change as we
# slightly alter each feature while holding others constant. We'll apply small
# perturbations to each feature across the dataset and monitor the variance in the predictions.

#Here's how we can implement this:

#Choose a perturbation level (typically a small percentage of the feature's standard deviation).
#Perturb each feature one at a time in the test dataset, and observe the change in predictions.
#Calculate the mean squared change in predictions for each feature to estimate its relative importance.

def feature_importance(gp, X, feature_names, epsilon=0.01):
    importances = np.zeros(X.shape[1])
    
    # Baseline prediction with original X
    baseline_pred = gp.predict(X)
    
    # Iterate over each feature
    for i in range(X.shape[1]):
        X_perturbed = np.copy(X)
        std_dev = np.std(X[:, i])
        
        # Perturb the feature
        X_perturbed[:, i] += epsilon * std_dev
        
        # Predict with perturbed data
        perturbed_pred = gp.predict(X_perturbed)
        
        # Calculate the mean squared difference
        importances[i] = np.mean((perturbed_pred - baseline_pred) ** 2)
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_feature_names = [feature_names[i] for i in indices]
    
    return sorted_importances, sorted_feature_names

# Feature names from the DataFrame
feature_names = X_train_df.columns.tolist()

# Calculate feature importances
sorted_importances, sorted_feature_names = feature_importance(gp, X_test, feature_names)

# Select the top 20 features
top_features = sorted_importances[:20]
top_feature_names = sorted_feature_names[:20]

# Plotting the top 20 feature importances
plt.figure(figsize=(12, 8))
plt.bar(top_feature_names, top_features)
plt.xticks(rotation=45, ha="right")
plt.xlabel('Feature')
plt.ylabel('Importance (Mean Squared Change)')
plt.title('Top 20 Feature Importance Estimated by Sensitivity Analysis')
plt.tight_layout()  # Adjust layout to make room for label rotation
plt.show()

#%%
# Define individual kernels
periodic_kernel = ExpSineSquared(length_scale=5.0, periodicity=24)
noise_kernel = WhiteKernel(noise_level=1)
smooth_kernel = RBF(length_scale=1.0)

# Example data
X_plot = np.linspace(0, 100, 100).reshape(-1, 1)

# Calculate covariance matrices for each kernel
K_periodic = periodic_kernel(X_plot)
K_noise = noise_kernel(X_plot)
K_smooth = smooth_kernel(X_plot)

# Plot each kernel's covariance matrix
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(K_periodic, interpolation='none', cmap='viridis')
plt.title('Periodic Kernel')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(K_noise, interpolation='none', cmap='viridis')
plt.title('Noise Kernel')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(K_smooth, interpolation='none', cmap='viridis')
plt.title('RBF Kernel')
plt.colorbar()

plt.show()


#%%

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(gp, X_train, y_train, cv=5, n_jobs=-1,
                                                        train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate mean and std deviation of training and test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')

plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()


#%%

# 1. Visualize Predictions vs True Values
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')

# 2. Plot Prediction Intervals
plt.subplot(1, 2, 2)
plt.errorbar(np.arange(len(y_test)), y_pred, yerr=1.96 * y_std, fmt='o', label='Predictions with 95% CI', alpha=0.5)
plt.scatter(np.arange(len(y_test)), y_test, color='red', label='True Values', alpha=0.7)
plt.xlabel('Data Point Index')
plt.ylabel('Predicted Value')
plt.title('Prediction Intervals')
plt.legend()

plt.tight_layout()
plt.show()

#%%
# Plotting the results with prediction intervals
plt.figure(figsize=(10, 5))

# Plot true values
plt.plot(np.arange(len(y_test)), y_test, 'r', label='True values', linewidth=2)

# Plot predicted values
plt.plot(np.arange(len(y_test)), y_pred, 'k', label='Predictions', linewidth=2)

# Plot the uncertainty (95% confidence interval)
plt.fill_between(np.arange(len(y_test)), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, alpha=0.2, color='gray', label='95% confidence interval')

plt.title('Gaussian Process Predictions with Periodic Kernel and Uncertainty')
plt.xlabel('Data Point')
plt.ylabel('Day-ahead Electricity Price')
plt.legend()
plt.show()






#%%

# %%
# Define the kernel
kernel = 1.0 * ExpSineSquared(length_scale=1.0, periodicity=24) + WhiteKernel(noise_level=1.0) + 1.0 * RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

param_grid = {
    "kernel__k1__k1__k2__length_scale": [0.1, 1.0, 10.0],  # Length scale for ExpSineSquared
    "kernel__k1__k1__k2__periodicity": [22, 24, 26],       # Periodicity for ExpSineSquared
    "kernel__k1__k2__noise_level": [0.1, 1.0, 10.0],       # Noise level for WhiteKernel
    "kernel__k2__k2__length_scale": [0.1, 1.0, 10.0]       # Length scale for RBF
}

# Setup and run the GridSearchCV
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Setup and run the GridSearchCV with verbose output
gscv = GridSearchCV(gp, param_grid, cv=5, verbose=3)  # Set verbose to 3 for more detailed output
gscv.fit(X_train, y_train)

# Output the best parameters
print("Best parameters:", gscv.best_params_)

# Output the best parameters
print("Best parameters:", gscv.best_params_)

# %%
