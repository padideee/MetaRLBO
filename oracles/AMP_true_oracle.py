
from oracles.base import BaseOracle
from torch.utils.data import DataLoader

class AMPTrueOracle(BaseOracle):
	def __init__(self, training_storage):
		self.training_storage = training_storage



	def query(self, model, x, flatten_input=True):
		"""
			Args:
			 - model: 
			 - x: (batch_size, dim of query)
			 - flatten_input: True if the model takes in the whole seq. at once (False o.w.)
			
			Return:
			 - Reward (Real Number): (batch_size, 1)
		"""
		if flatten_input: 
			x = x.flatten(start_dim=-2, end_dim = -1) # Hardcoded for AMP

		return model.predict_proba(x)


	def fit(self, model, flatten_input=True):
		"""
			Fits the model on the entirety of the storage

		"""

		if flatten_input:
			seq = self.training_storage.mols.flatten(start_dim=-2, end_dim=-1) # Hardcoded for AMP
		else:
			seq = self.training_storage.mols
		value = self.training_storage.scores

		return model.fit(seq, value)



