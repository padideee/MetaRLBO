

class AMPProxyOracle(BaseOracle):

	def __init__(self, query_storage):
		super(BaseOracle)
		self.query_storage = query_storage



	def query(self, model, mols):
		raise NotImplementedError()

	def fit(self, model):
		"""
			Fits the model on a randomlmy sampled subset of the storage.

		"""
		raise NotImplementedError()