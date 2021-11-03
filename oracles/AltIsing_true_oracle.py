from oracles.base import BaseOracle
from torch.utils.data import DataLoader
import numpy as np


class AltIsingTrueOracle(BaseOracle):
	def __init__(self):
		self.queried_scores = {}
		self.query_count = 0


	def query(self, model, x, flatten_input=True):
		"""
			Args:
			 - model: 
			 - x: (batch_size, dim of query) (Assumption: No duplicates within a batch!)
			 - flatten_input: True if the model takes in the whole seq. at once (False o.w.)
			
			Return:
			 - Reward (Real Number): (batch_size, 1)
		"""
		if flatten_input: 
			pass # Not useful for Ising

		batch_size = x.shape[0]
		pred_prob = np.zeros(batch_size)

		to_query = []

		for i in range(batch_size):
			tuple_xi = tuple(x[i].argmax(-1).tolist())

			if tuple_xi in self.queried_scores:
				pred_prob[i] = self.queried_scores[tuple_xi]
				
				print("Duplicate")
			else:
				to_query.append(i)


		pred_prob[to_query] = model(x[to_query].argmax(-1))

		for i in range(batch_size):
			tuple_xi = tuple(x[i].argmax(-1).tolist())

			if tuple_xi not in self.queried_scores:
				self.queried_scores[tuple_xi] = pred_prob[i]
				self.query_count += 1


		return pred_prob


	def fit(self, model, flatten_input=True):
		"""
			Fits the model on the entirety of the storage

		"""

		return model

