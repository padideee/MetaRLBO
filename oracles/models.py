from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, BayesianRidge

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern
from oracles.custom_models.GPR import CustomGPR
from oracles.custom_models.dynappo_ensemble import DynaPPOEnsemble
from oracles.custom_models.alt_ising_model import AlternatingChainIsingModel
from oracles.custom_models.NN_model import NNProxyModel
from oracles.custom_models.rna_model import RNAModel
# Classifiers

def RFC(n_estimators=200):
    model = RandomForestClassifier(n_estimators=n_estimators)
    return model


def NN():
    raise NotImplementedError()



# Regressors

def RFR(max_depth = 50, max_features = "log2", n_estimators = 100):
    model = RandomForestRegressor(max_depth = max_depth,
                                  max_features = max_features,
                                  n_estimators = n_estimators)
    return model


def KNR(n_neighbors = 5, metric="hamming"):
    model = KNeighborsRegressor(n_neighbors=n_neighbors) # Temporarily hardcoded
    return model

def RR():
    model = Ridge(alpha=1.0)
    return model

def BR(alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06):
  model = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1 = lambda_1, lambda_2 = lambda_2)
  return model


def XGBoost(learning_rate = 0.1, max_depth = 85, n_estimators = 500):
    model = GradientBoostingRegressor(n_estimators = n_estimators,
                                      max_depth = max_depth,
                                      learning_rate = learning_rate)
    return model

def XGB():
    model = xgb.XGBRegressor(objective ='reg:squarederror',
                             colsample_bytree = 0.3,
                             learning_rate = 0.1,
                             max_depth = 85, alpha = 5)
    return model


def GPR(kernel = "RBF", random_proj=True, embedding_size=25):
    if kernel == "RBF":
        kernel_model = RBF()
    elif kernel == "RationalQuadratic":
        kernel_model = RationalQuadratic()
    elif kernel == "Matern":
        kernel_model = Matern()
    else:
        raise NotImplementedError

    model = CustomGPR(kernel=kernel_model, random_proj=random_proj, embedding_size=embedding_size)

    return model

def Ensemble():
    # GPR(kernel="RBF", random_proj=False)... or other kernels
    return DynaPPOEnsemble([KNN(), RFR(), XGB(), XGBoost(), GPR(kernel="RBF", random_proj=False), GPR(kernel="RationalQuadratic", random_proj=False), GPR(kernel="Matern", random_proj=False)])



def AltIsingModel(length=50, vocab_size=20):
    return AlternatingChainIsingModel(length=length, vocab_size=vocab_size)


def MLP(seq_len, alphabet_len, device = None): # TODO: Setup so we can use the device...
  if device is None:
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  return NNProxyModel(seq_len=seq_len, alphabet_len=alphabet_len, model_type="MLP", device=device)

def CNN(seq_len, alphabet_len, device = None): # TODO: Setup so we can use the device...
  if device is None:
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  return NNProxyModel(seq_len=seq_len, alphabet_len=alphabet_len, model_type="CNN", device=device)

def CNNdropout(seq_len, alphabet_len, device = None):
    if device is None:
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return NNProxyModel(seq_len=seq_len, alphabet_len=alphabet_len, model_type="CNN_dropout", device=device)

def RNA_Model():
  return RNAModel()
