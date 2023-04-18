import matplotlib.pyplot as plt
import pandas as pd
import pprint
import numpy as np
import json


try:
    import flexs
    from flexs import baselines
    import flexs.utils.sequence_utils as s_utils
except:
    print("Need to setup the conda environment for RNA14")


class RNAModel:

    def __init__(self, ptype="L14_RNA1"):

        self.problem = flexs.landscapes.rna.registry()[ptype]
        pprint.pprint(self.problem)

        self.landscape = flexs.landscapes.RNABinding(**self.problem['params'])


        self.alphabet = s_utils.RNAA
        self.char_to_int = {}
        self.int_to_char = {}

        for i, c in enumerate(self.alphabet):
            self.char_to_int[c] = i
            self.int_to_char[i] = c
        

    def __call__(self, seqs):
        """ 
            seqs: [nSequences, seq_length] (not one hot encoded)
        """

        all_str_seqs = []
        for i in range(seqs.shape[0]):
            enc_str = ""
            for j in range(seqs.shape[1]):
                enc_str = enc_str + self.int_to_char[seqs[i][j].item()]
            all_str_seqs.append(enc_str)

        """
            Recall landscape get_fitness requires:
            (Union[List[str], ndarray]) – A list/numpy array of sequence strings to be scored.
        """

        return self.landscape.get_fitness(all_str_seqs) # returns score for each seq