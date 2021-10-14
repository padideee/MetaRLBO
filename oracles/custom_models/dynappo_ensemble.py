from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold
import numpy as np

class DynaPPOEnsemble(RegressorMixin):
    def __init__(self, poss_models, **kwargs):
        super().__init__(**kwargs)
        self.possible_models = poss_models
        self.models = []

    def fit(self, X, y, **kwargs):
        def kFoldCV(models, X, y, k=5):
            """
            Perform k-fold validation, return average score
            """
            kf = KFold(n_splits=k,  shuffle=True) #TODO
            kf.get_n_splits(X)

            accuracy = np.zeros((k, len(models)))
            i = 0
            for train_index, test_index in kf.split(X):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]

                c = regressor(X_train, np.squeeze(y_train))

                for n, m in enumerate(models):
                    accuracy[i, n] = c.get_accuracy(c.to_fit(m), X_test, np.squeeze(y_test))
                i += 1
            print("Accuracy array")
            print(accuracy)
            return np.average(accuracy, axis = 0)

        def test_train(models, X, y, test_size = 0.3, shuffle = True):
            self.proxy.proxylist = []
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0, shuffle=shuffle)

            print("Shape: ", X_train.shape, y_train.shape, X_test.shape)

            c = regressor(X_train, y_train)

            for n, m in enumerate(models):
                pr = c.to_fit(m)
                ac = c.get_accuracy(pr, X_test, y_test)
                self.proxy.proxylist.append(Reward(pr, ac))

            return self.proxy.get_best_proxy()

        models = self.possible_models

        if kfld:
            accuracy = kFoldCV(models=models, X=X, y=y)
            print("Final accuracy: ", accuracy)
            print(accuracy > 0.5)

            # Select all proxy models with accuracy > threshold
            proxy = models[accuracy > 0.5]
            self.models = []
            for p in proxy:
                # Tune regressor on all data seen so far
                self.models.append(regressor(X, y).to_fit(p))
        else:

            self.models = [test_train(models=models, X=X, y=y)]


    def predict(self, X, **kwargs):
        predictions = [self.models[i].predict(X, y, **kwargs) for i in range(len(self.models))]
        return sum(predictions) / len(predictions)

    