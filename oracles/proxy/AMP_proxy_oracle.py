from torch.utils.data import DataLoader
from oracles.base import BaseOracle

class AMPProxyOracle(BaseOracle):
	def __init__(self, training_storage, p = 0.8):
		self.training_storage = training_storage

		self.p = p


	def query(self, model, x, flatten_input=True):
		if flatten_input:
			x = x.flatten(start_dim=-2, end_dim = -1)
		return model.predict_proba(x)

	def fit(self, model, flatten_input=True):
		"""
			Fits the model on a randomly sampled (p) subset of the storage.

			TODO
		"""

		if flatten_input:
			seq = self.training_storage.mols.flatten(start_dim=-2, end_dim=-1)
		else:
			seq = self.training_storage.mols
		value = self.training_storage.scores

		return model.fit(seq, value)