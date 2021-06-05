


class TrueOracle(BaseOracle):
	def __init__(self, storage):
		self.storage = storage


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
			Fits the model on the entirety of the storage

			Args:
			 - model: 

		"""

		raise NotImplementedError()
