from oracles.base import BaseOracle
from torch.utils.data import DataLoader
import numpy as np


class AMPTrueOracle(BaseOracle):
	def __init__(self, training_storage):
		self.training_storage = training_storage
		self.queried_scores = {}
		self.query_count = 0


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



		batch_size = x.shape[0]
		pred_prob = np.zeros((batch_size, 2))

		for i in range(batch_size):


			if tuple(x[i]) in self.queried_scores:
				pred_prob[i] = self.queried_scores[tuple(x[i])]
				# print("Same Query Count")
			else:
				# Leo: Should be parallelised
				score = model.predict_proba(x[i][np.newaxis, ...])
				self.query_count += 1
				# print("+1 Query Count")

				if score.shape[-1] == 1:
					score = np.zeros((2, ))

					assert model.classes_.shape[-1] <= 2
					
					if model.classes_[0] == 0:
						score[0] = 1
					elif model.classes_[0] == 1:
						score[1] = 1
					else:
						raise NotImplementedError
				pred_prob[i] = score

			



		# pred_prob = model.predict_proba(x)

		# assert model.classes_.shape[-1] <= 2
		# # Special case (Only a single class):
		# if pred_prob.shape[-1] == 1:
		# 	pred_prob = np.zeros((*pred_prob.shape[:-1], 2))

		# 	if model.classes_[0] == 0:
		# 		pred_prob[:, 0] = 1
		# 	elif model.classes_[0] == 1:
		# 		pred_prob[:, 1] = 1


		return pred_prob[:, 1]


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



