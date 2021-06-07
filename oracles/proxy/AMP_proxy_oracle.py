

class AMPProxyOracle(BaseOracle):
	def __init__(self, query_storage, p = 0.8):
		super(BaseOracle)
		self.query_storage = query_storage

		self.p = p


	def query(self, model, mols):
		raise NotImplementedError()

	def fit(self, model):
		"""
			Fits the model on a randomly sampled (p) subset of the storage.

		"""
		raise NotImplementedError()