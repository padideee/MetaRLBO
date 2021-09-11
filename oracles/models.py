from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, BayesianRidge
import xgboost as xgb

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern

# Classifiers

def RFC(n_estimators=200):
    model = RandomForestClassifier(n_estimators=n_estimators)
    return model


def NN():
    raise NotImplementedError()



# Regressors

def RFR(max_depth = 50, max_features = "log2", n_estimators = 100):
    model = RandomForestRegressor(max_depth = max_depth,
                                  max_features = max_features,
                                  n_estimators = n_estimators)
    return model


def KNR(n_neighbors = 5, metric="hamming"):
    model = KNeighborsRegressor(n_neighbors=n_neighbors) # Temporarily hardcoded
    return model

def RR():
    model = Ridge(alpha=1.0)
    return model

def BR(alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06):
  model = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1 = lambda_1, lambda_2 = lambda_2)
  return model


def XGBoost(learning_rate = 0.1, max_depth = 85, n_estimators = 500):
    model = GradientBoostingRegressor(n_estimators = n_estimators,
                                      max_depth = max_depth,
                                      learning_rate = learning_rate)
    return model

def XGB():
    model = xgb.XGBRegressor(objective ='reg:squarederror',
                             colsample_bytree = 0.3,
                             learning_rate = 0.1,
                             max_depth = 85, alpha = 5)
    return model


def GPR(kernel_type = "RBF"):
    if kernel_type == "RBF":
        kernel = RBF()
    elif kernel_type == "RationalQuadratic":
        kernel = RationalQuadratic()
    elif kernel_type == "Matern":
        kernel = Matern()
    else:
        raise NotImplementedError
    model = GaussianProcessRegressor(kernel=kernel)

    return model
