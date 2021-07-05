from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

def RFC():
	model = RandomForestClassifier(random_state=0, bootstrap= True, max_depth=50, n_estimators=200)
	return model


def NN():
	raise NotImplementedError()


def RFR():
	model = RandomForestRegressor(random_state=0, bootstrap= True, max_depth=10, n_estimators=200)
	return model

def KNR():
	model = KNeighborsRegressor(n_neighbors=7)
	return model