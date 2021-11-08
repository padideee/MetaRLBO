import torch
from storage.base import BaseStorage

class QueryStorage(BaseStorage):
    def __init__(self, storage_size, state_dim):
        super().__init__(storage_size)

        self.state_dim = state_dim


        self.mols = torch.zeros(self.storage_size, *state_dim)
        self.scores = torch.zeros(self.storage_size)

        self.mols_set = set([""]) # Initialise with the empty molecule...         

        self.storage_filled = 0

        self.curr_idx = 0




    def insert(self, x, y):
        """
            Performs checks for duplicates. Inserts data into the storage in a queue like fashion [cur_idx, cur_idx + batch_size]
            
            args:
             - x: Torch.tensor of size (batch_size, state_dim)
             - y: (batch_size, )

        """

        # raise NotImplementedError()

        # Check for duplicates
        valid_idx = [] # Valid only if unique
        for i in range(x.shape[0]):
            from data.dynappo_data import enc_to_seq
            seq = enc_to_seq(x[i])
            if seq.find(">") != -1:
                seq = seq[:seq.find(">")]
            if seq not in self.mols_set:
                valid_idx.append(i)
                self.mols_set.add(seq)
            else:
                print("Storage: Already queried...", seq)

        x = x[valid_idx]
        y = y[valid_idx]


        batch_size = x.shape[0]

        if self.curr_idx + batch_size <= self.storage_size:
            self.mols[self.curr_idx : self.curr_idx + batch_size] = x
            self.scores[self.curr_idx : self.curr_idx + batch_size] = y
        else:

            self.mols[self.curr_idx : self.storage_size] = x[: self.storage_size - self.curr_idx]
            self.scores[self.curr_idx : self.storage_size] = y[: self.storage_size - self.curr_idx]

            self.mols[: batch_size - (self.storage_size - self.curr_idx)] = x[self.storage_size - self.curr_idx:]
            self.scores[: batch_size - (self.storage_size - self.curr_idx)] = y[self.storage_size - self.curr_idx:]




        self.curr_idx = (self.curr_idx + batch_size) % self.storage_size 
        self.storage_filled = min(self.storage_filled + batch_size, self.storage_size)