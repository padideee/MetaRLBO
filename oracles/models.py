from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

# Classifiers

def RFC():
	model = RandomForestClassifier(random_state=0, bootstrap= True, max_depth=50, n_estimators=200)
	return model


def NN():
	raise NotImplementedError()



# Regressors

def RFR(max_depth = 50, max_features = "auto", n_estimators = 100):
    model = RandomForestRegressor(max_depth = max_depth,
                                  max_features = max_features,
                                  n_estimators = n_estimators)
    return model


def KNR(n_neighbors = 10):
	model = KNeighborsRegressor(n_neighbors=neighbors) # Temporarily hardcoded
	return model

def RR():
	model = Ridge(alpha=1.0)
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
