from data.process_data import get_AMP_data
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



class get_test_oracle:
    """
    The oracle fitted on test set
    TODO: move this to oracle folder
    TODO: other method for evaluation
    """
    def __init__(self):

        # path to pickle format instead: 'data/data_test.pickle'
        data_storage = get_AMP_data('data/data_test.pickle')  # our held-out AMP for training the classifier for evaluation


        seq, label = data_storage.mols, data_storage.scores


        seq = np.array(seq)
        label = np.array(label)
        n, x, y = seq.shape
        seq = seq.reshape((n, x * y))

        self.model_test = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=50, n_estimators=200)
        self.model_test.fit(seq, label.flatten())


    # def give_score(self, mols, scores):
    #     labels = (scores >= 0.5).float()
    #     seqs = mols.flatten(1, 2)

    #     return self.model_test.score(seqs, labels)

    def get_prob(self, mols):
        seqs = mols.flatten(1, 2)

        prob = self.model_test.predict_proba(seqs)

        return prob[:, 1]
