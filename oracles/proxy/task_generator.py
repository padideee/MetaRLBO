from sklearn.utils import resample
import numpy as np


class task_generator:
    def __init__(self, seq, label, n_tasks, seed, task_method='bootstrap'):
        self.seq = seq
        self.label = label
        self.n_tasks = n_tasks
        self.seed = seed
        self.task_method = task_method

        if task_method == 'bootstrap':
            self.idx = bootstrap(range(len(self.seq)), n_samples=self.n_tasks, random_state=self.seed)


    def get_task(self):
        seq_k = np.take(self.seq, self.idx, axis=0)
        label_k = np.take(self.label, self.idx, axis=0)

        return seq_k, label_k


def bootstrap(x, n_samples, random_state=None):
    return resample(x, replace=True, n_samples=n_samples, random_state=random_state)