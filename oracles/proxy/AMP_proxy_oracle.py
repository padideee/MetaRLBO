from torch.utils.data import DataLoader
from oracles.base import BaseOracle

class AMPProxyOracle(BaseOracle):
	def __init__(self, training_storage, p = 0.8):
		super(BaseOracle)
		self.training_storage = training_storage

		self.p = p


	def query(self, model, mols):
		return self.model.predict_proba(mols)

	def fit(self, model):
		"""
			Fits the model on a randomly sampled (p) subset of the storage.
		"""

		seq = self.training_storage.mols
		value = self.training_storage.scores

		return self.model.fit(seq, value)