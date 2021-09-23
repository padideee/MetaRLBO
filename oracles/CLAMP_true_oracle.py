from oracles.base import BaseOracle
from torch.utils.data import DataLoader
import numpy as np

from data.dynappo_data import enc_to_seq


import torch


class CLAMPTrueOracle(BaseOracle):
	def __init__(self):
		self.query_count = 0

	def query(self, model, x, flatten_input=False):
		"""
			Args:
			 - model: 
			 - x: (batch_size, dim of query)
			 - flatten_input: True if the model takes in the whole seq. at once (False o.w.)
			
			Return:
			 - Reward (Real Number): (batch_size, 1)
		"""
		
		batch_size = x.shape[0]
		seqs = []
		for i in range(batch_size):
			seq = enc_to_seq(x[i])

			seq = seq[:seq.find(">")]
			seqs.append(seq)

		print(seqs)
		self.query_count += batch_size

		return model.evaluate_many(seqs)["confidence"][:, 1]



	def fit(self, model, flatten_input=False):
		"""
			Fits the model on the entirety of the storage (unneeded since CLAMP oracle is already trained...)

		"""

		return model



