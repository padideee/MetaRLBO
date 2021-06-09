from sklearn.ensemble import RandomForestClassifier


class BaseOracle:
	def __init__(self, model_name='RFC'):

		if model_name == 'RFC':
			self.model = RFC()
		elif model_name == 'NN':
			self.model = NN()


		pass

	def query(self,  x):
		"""
			Args:
			 - model: 
			 - x: (batch_size, dim of query)
			
			Return:
			 - Reward (Real Number): (batch_size, 1)
		"""
		# raise NotImplementedError()

		return self.model.predict_proba(x)



	def fit(self, seq, value):
		"""
			Fits the model on the storage

			Args:
			 - model: 

			Return:
			 - Trained Model

		"""

		# raise NotImplementedError()
		return self.model.fit(seq, value)


def RFC():
	model = RandomForestClassifier(random_state=0, bootstrap= True, max_depth=50, n_estimators=200)
	return model


def NN():
	raise NotImplementedError()
