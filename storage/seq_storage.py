import torch
from storage.base import BaseStorage

class SeqStorage(BaseStorage):
    def __init__(self):
        super().__init__(None)

        self.mol_strings = [] # Stores the mol string!
        self.mol_round = [] # Stores which round it comes from
        self.mol_query_proxy_idx = [] # Stores which query proxy it comes from


    def insert(self, mols, query_proxy_idx, query_round=0):
        """
            Performs checks for duplicates. Inserts data into the storage in a queue like fashion [cur_idx, cur_idx + batch_size]
            
            args:
             - x: Torch.tensor of size (batch_size, state_dim)
             - y: (batch_size, )

        """
        self.mol_strings = self.mol_strings + mols
        self.mol_round = self.mol_round + query_round
        self.mol_query_proxy_idx = self.mol_query_proxy_idx + query_proxy_idx
