from torch.utils.data import DataLoader
from oracles.base import BaseOracle
import torch
import numpy as np

class AMPProxyOracle(BaseOracle):
	def __init__(self, training_storage, p = 0.8):
		self.training_storage = training_storage

		self.p = p


	def query(self, model, x, flatten_input=True, **kwargs):

		"""
			Args:
			  - (batch_size, ...)
		
			Returns:
			  - (batch_size, )
		"""
		if flatten_input:
			x = x.flatten(start_dim=-2, end_dim = -1)

		return model.predict(x, **kwargs) # Regressor 

	def fit(self, model, flatten_input=True):
		"""
			Fits the model on a randomly sampled (p) subset of the storage.
		"""

		if flatten_input:
			seq = self.training_storage.mols.flatten(start_dim=-2, end_dim=-1)
		else:
			seq = self.training_storage.mols
		value = self.training_storage.scores


		# Randomly sample subset of storage
		size = self.training_storage.storage_filled
		indices = torch.tensor(np.random.choice(size, int(self.p*size), replace=False))

		sampled_seq = torch.index_select(seq, dim=0, index=indices)
		sampled_value = torch.index_select(value, dim=0, index=indices)


		return model.fit(sampled_seq, sampled_value)