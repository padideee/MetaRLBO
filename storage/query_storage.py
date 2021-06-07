"""
	Storage for training the oracles



	Implement a Queue (as a tensor):
	0...storage_size, 0, ...
"""

import torch

class QueryStorage(BaseStorage):
	def __init__(self, storage_size, state_dim):
		super().__init__(storage_size)

		mols = torch.tensor(self.storage_size, state_dim)
		scores = torch.tensor(self.storage_size, 1)
		




	def insert(self, x, y):
		"""
			Inserts data into the storage in a queue like fashion [cur_idx, cur_idx + batch_size]
			
			args:
			 - x: (batch_size, state_dim)
			 - y: (batch_size, 1)

		"""
		raise NotImplementedError()