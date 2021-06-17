
def RFC():
	model = RandomForestClassifier(random_state=0, bootstrap= True, max_depth=50, n_estimators=200)
	return model


def NN():
	raise NotImplementedError()