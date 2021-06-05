"""
	Storage for training the policy



	Implement a Queue (as a tensor):
	0...storage_size, 0, ...
"""


class RolloutStorage(BaseStorage):

	def __init__(self):
		raise NotImplementedError()



	def insert(self, x, y):
		raise NotImplementedError()