

def get_true_oracle_model(config):
	"""
		Returns an instance of TrueOracle following config
	"""

	raise NotImplementedError()

	
def get_proxy_oracle_model(config):
	"""
		Returns an instance of (AMP)ProxyOracle following config.

		Note that we should be allowed to have different kinds of the proxy oracle depending on ours configs.
	"""
	raise NotImplementedError()

