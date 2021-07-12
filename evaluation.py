from data.process_data import get_data
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



class get_test_proxy:
    """
    The oracle fitted on test set
    TODO: move this to oracle folder
    TODO: other method for evaluation
    """
    def __init__(self, i):
        self.i = i

        # path to pickle format instead: 'data/data_test.pickle'
        seq, label = get_data('data/data_test.hkl')  # our held-out AMP for training the classifier for evaluation

        seq = np.array(seq)
        label = np.array(label)
        n, x, y = seq.shape
        seq = seq.reshape((n, x * y))

        self.model_test = RandomForestClassifier(random_state=0, bootstrap=True, max_depth=50, n_estimators=200)
        self.model_test.fit(seq, label)

        self.df = pd.read_pickle('logs/D3.pkl')  # D3 contains the generated AMP


    def give_class_label(self):
        pred_prob = self.df['pred_prob'].values
        label = []
        for i in pred_prob:
            if i >= 0.5:
                label.append('positive')
            else:
                label.append('negative')
        return label

    def give_seqs(self):
        seq = self.df['embed_seq'].values
        seqs = []
        for i in seq:
            seqs.append(i[0])
        return seqs

    def give_score(self):
        labels = self.give_class_label()
        seqs = self.give_seqs()

        return self.model_test.score(seqs, labels)

