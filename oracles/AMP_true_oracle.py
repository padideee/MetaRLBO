from oracles.base import BaseOracle
from torch.utils.data import DataLoader
import numpy as np


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

		pred_prob = model.predict_proba(x)


		assert model.classes_.shape[-1] <= 2
		# Special case (Only a single class):
		if pred_prob.shape[-1] == 1:
			pred_prob = np.zeros((*pred_prob.shape[:-1], 2))

			if model.classes_[0] == 0:
				pred_prob[:, 0] = 1
			elif model.classes_[0] == 1:
				pred_prob[:, 1] = 1


		return pred_prob


	def fit(self, model, flatten_input=True):
		"""
			Fits the model on the entirety of the storage

		"""

		if flatten_input:
			seq = self.training_storage.mols.flatten(start_dim=-2, end_dim=-1) # Hardcoded for AMP
		else:
			seq = self.training_storage.mols
		value = self.training_storage.scores.flatten() # [batch_size, 1] -> [batch_size]

		return model.fit(seq, value)



