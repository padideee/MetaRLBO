import torch
from storage.base import BaseStorage

class QueryStorage(BaseStorage):
	def __init__(self, storage_size, state_dim):
		super().__init__(storage_size)

		self.state_dim = state_dim


		self.mols = torch.zeros(self.storage_size, *state_dim, 21)
		self.scores = torch.zeros(self.storage_size)
		


		self.curr_idx = 0




	def insert(self, x, y):
		"""
			Inserts data into the storage in a queue like fashion [cur_idx, cur_idx + batch_size]
			
			args:
			 - x: (batch_size, state_dim)
			 - y: (batch_size, 1)

		"""

		# raise NotImplementedError()


		batch_size = x.shape[0]

		if self.curr_idx + batch_size <= self.storage_size:
			self.mols[self.curr_idx : self.curr_idx + batch_size] = x
			self.scores[self.curr_idx : self.curr_idx + batch_size] = y
		else:

			self.mols[self.curr_idx : self.storage_size] = x[: self.storage_size - self.curr_idx]
			self.scores[self.curr_idx : self.storage_size] = y[: self.storage_size - self.curr_idx]

			self.mols[: batch_size - (self.storage_size - self.curr_idx)] = x[self.storage_size - self.curr_idx:]
			self.scores[: batch_size - (self.storage_size - self.curr_idx)] = y[self.storage_size - self.curr_idx:]




		self.curr_idx = (self.curr_idx + batch_size) % self.storage_size 