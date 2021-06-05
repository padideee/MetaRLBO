


class BaseOracle:
	def __init__(self):

		pass

	def query(self, model, x):
		"""
			Args:
			 - model: 
			 - x: (batch_size, dim of query)
			
			Return:
			 - Reward (Real Number): (batch_size, 1)
		"""
		raise NotImplementedError()


	def fit(self, model):
		"""
			Fits the model on the storage

			Args:
			 - model: 

			Return:
			 - Trained Model

		"""

		raise NotImplementedError()