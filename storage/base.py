


class BaseStorage:
	def __init__(self, storage_size):

		self.storage_size = storage_size



	def insert(self, x, y):
		raise NotImplementedError()