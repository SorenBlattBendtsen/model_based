#%%
import pandas as pd
import numpy as np
from prepare_country_data import transpose_for_country_code
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/2023/nordic_energy_data.csv")
df_dk1 = transpose_for_country_code(df, "DK_1")  

# drop columns in df_dk1 where there are only 0 values or nan in every row
df_dk1 = df_dk1.loc[:, (df_dk1 != 0).any(axis=0)]
# remove Cap_to_SE_4_FI
#df_dk1 = df_dk1.drop(columns=["Cap_to_SE_4_FI", "country_code"])

# Make X_train and y_train, and X_test, y_test
#select only first 100 rows in df_dk1
df_dk1 = df_dk1.iloc[:2000]
# Separate features (X) and target variable (Y)
X = df_dk1.drop(columns=["DA-price [EUR/MWh]", "Timestamp"])  
Y = df_dk1["DA-price [EUR/MWh]"]
N = len(X)
N_train = int(0.8*N)
N_test = N - N_train
X_indices = X.index
timestamps = df_dk1.loc[X_indices]['Timestamp']
#%%
# Standardize the data
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)
Y = pd.DataFrame(scaler_Y.fit_transform(Y.values.reshape(-1,1)), columns=["DA-price [EUR/MWh]"])


# Split data into training set and test set (80% training and 20% testing)
X_train = X.iloc[:N_train]
X_test = X.iloc[N_train:]
y_train = Y.iloc[:N_train]
y_test = Y.iloc[N_train:]

# To numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()





#%%
cov_params = [0.001]
def covSE(x1, x2, cov_params):
    return np.exp(-cov_params[0]*np.sum((x1 - x2)**2))

def covPER(x1, x2, cov_params):
    return cov_params[0] * np.exp(-cov_params[1] * np.sum( np.sin(np.pi*(x1-x2)/cov_params[2])**2 ))

def covWN(x1, x2, cov_params):
    if x1 == x2:
        return cov_params[0]
    else:
        return 0
    
def covSUM_PER_WN(x1, x2, cov_params):
    return covPER(x1, x2, cov_params[:3]) + covWN(x1, x2, cov_params[3:])

# Covariance matrix calculater:
def cov(x1, x2, cov_fn, cov_params):
    K = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            K[i,j] = cov_fn(x1[i,:], x2[j,:], cov_params)
    return K

# Sample from GP prior
def sample_GP_prior(X, N_samples, cov_func, cov_params):

    # construct K
    K = cov(X, X, cov_func, cov_params)
    plt.imshow(K)
    plt.title("Covariance matrix from Gaussian process prior")

    # take 5 samples from the prior multivariate Gaussian: Normal(0, K)
    y = np.random.multivariate_normal(np.zeros(len(K)), K, N_samples)

    # plot samples
    plt.figure()
    for i in range(len(y)):
        plt.plot(X, y[i])
    plt.title("%d random functions sampled from the Gaussian process prior" % (len(y),))
    plt.show()


# specify inputs
X_prior = np.arange(-4.0,4.0,0.05)[:,np.newaxis]

sample_GP_prior(X_prior, 5, covSE, cov_params)


def compute_predictive_posterior(X_test, sigma, cov_params):
    N_test = len(X_test)
    
    K_sigma = cov(X_train, X_train, covSE, cov_params) + sigma**2 * np.eye(N_train)
    K_sigma_inv = np.linalg.inv(K_sigma)

    predictions = np.zeros(N_test)
    variances = np.zeros(N_test)
    for i in range(N_test):
        xstar = X_test[i,:][:,np.newaxis] # the test point x*

        # compute k(X,x*)
        K_xstar = cov(X_train, xstar, covSE, cov_params)

        # compute k(x*,x*)
        K_xstar_xstar = cov(xstar, xstar, covSE, cov_params)

        # make prediction
        # <y*> = k(x*,X) (K + \sigma^2 I)^{-1} Y
        ystar_mean = np.dot(np.dot(K_xstar.T, K_sigma_inv), y_train)

        # compute prediction variance
        # var(y*)^2 = k(x*,x*) + \sigma^2 - k(X,x*).T (K + \sigma^2 I)^{-1} k(X,x*)
        ystar_var = K_xstar_xstar + sigma**2 - np.dot(np.dot(K_xstar.T, K_sigma_inv), K_xstar)

        predictions[i] = ystar_mean[0,0]
        variances[i] = ystar_var[0,0]
    
    return predictions, variances


cov_params = [0.001]
sigma = 0.1


predictions, variances = compute_predictive_posterior(X_test, sigma, cov_params)


# %%
#make a plot of the predictions and the actual values
plt.plot(timestamps.iloc[N_train:], y_test, 'ro')
plt.plot(timestamps.iloc[N_train:], predictions, "bo")
plt.legend(["Actual values", "Predictions"])
plt.show()

#calculate the rms
rms = np.sqrt(np.mean((predictions - y_test)**2))
print("RMS: %.3f" % rms)


# %%

def log_marginal_loglikelihood(params):
    K_sigma = cov(X_train, X_train, covPER, params) + sigma**2 * np.eye(N_train)
    K_sigma_inv = np.linalg.inv(K_sigma)

    loglikelihood = -0.5*np.dot(np.dot(y_train.T, K_sigma_inv), y_train) - 0.5*np.log(np.linalg.det(K_sigma)) - 0.5*N_train*np.log(2*np.pi)
    print("loglikelihood: %.3f\t(params=%s)" % (loglikelihood[0,0], str(params)))
    
    return -loglikelihood[0]

def covSEiso_gradient(params):
    W = np.zeros((N_train, N_train))
    for i in range(N_train):
        for j in range(N_train):
            W[i,j] = -np.sum((X_train[i,:] - X_train[j,:])**2) * np.exp(-params[0]*np.sum((X_train[i,:] - X_train[j,:])**2))
    return W

def gradient(params):
    gradient = np.zeros(len(params))
    Wl = covSEiso_gradient(params)
    
    K_sigma = cov(X_train, X_train, covPER, params) + sigma**2 * np.eye(N_train)
    K_sigma_inv = np.linalg.inv(K_sigma)
    K_inv_Wl = np.dot(K_sigma_inv, Wl)
    
    gradient[0] = 0.5*np.dot(np.dot(np.dot(y_train.T, K_inv_Wl), K_sigma_inv), y_train) - 0.5*np.trace(K_inv_Wl)
    return -gradient

from scipy.optimize import fmin_cg
cov_params_init = cov_params
cov_params_opt = fmin_cg(log_marginal_loglikelihood, cov_params_init, fprime=gradient, maxiter=10)

predictions, variances = compute_predictive_posterior(X_test, sigma, cov_params_opt)

#%%
#make a plot of the predictions and the actual values with optimized parameters
plt.plot(timestamps.iloc[N_train:], y_test, 'ro')
plt.plot(timestamps.iloc[N_train:], predictions, "bo")
plt.legend(["Actual values", "Predictions"])
plt.show()
# %%
