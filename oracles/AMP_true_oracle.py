
from oracles.base import BaseOracle
from torch.utils.data import DataLoader

class AMPTrueOracle(BaseOracle):
	def __init__(self, training_storage, model_name):
		self.training_storage = training_storage



	def query(self, model, x):
		"""
			Args:
			 - model: 
			 - x: (batch_size, dim of query)
			
			Return:
			 - Reward (Real Number): (batch_size, 1)
		"""

		return self.model.predict_proba(x)


	def fit(self, model):
		"""
			Fits the model on the entirety of the storage

		"""

		seq = self.training_storage.mols
		value = self.training_storage.scores

		return self.model.fit(seq, value)



