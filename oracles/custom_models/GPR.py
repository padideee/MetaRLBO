from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np



class CustomGPR(GaussianProcessRegressor):
	random_mat_generated = False
	random_mat = None
	div_weight = 1
	def __init__(self, random_proj=True, embedding_size=25, **kwargs):
		super().__init__(**kwargs)
		self.random_proj = random_proj
		self.embedding_size = embedding_size

	def fit(self, X, y, **kwargs):
		if self.random_proj:
			if not CustomGPR.random_mat_generated:
				CustomGPR.random_mat_generated = True
				CustomGPR.random_mat = np.random.rand(X.shape[-1], self.embedding_size)
				# CustomGPR.div_weight = (X @ CustomGPR.random_mat).mean()


			return super().fit(X @ CustomGPR.random_mat / CustomGPR.div_weight, y, **kwargs)
		else:
			return super().fit(X, y, **kwargs)

	def predict(self, X, **kwargs):
		if self.random_proj:
			return super().predict(X @ CustomGPR.random_mat, **kwargs)
		else:
			return super().predict(X, y, **kwargs)

    