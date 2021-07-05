from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge


def RFC():
	model = RandomForestClassifier(random_state=0, bootstrap= True, max_depth=50, n_estimators=200)
	return model


def NN():
	raise NotImplementedError()


def RFR():
	model = RandomForestRegressor(random_state=0, bootstrap= True, max_depth=10, n_estimators=200)
	return model

def KNR():
	model = KNeighborsRegressor(n_neighbors=3) # Temporarily hardcoded
	return model

def RR():
	model = Ridge(alpha=1.0)
	return model