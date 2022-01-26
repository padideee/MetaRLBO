#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flexs


# In[1]:


import editdistance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pprint
import numpy as np
import json

import flexs
from flexs import baselines
import flexs.utils.sequence_utils as s_utils
import torch.nn.functional as F
import torch


# In[2]:


exp_name = 'AMP'

# Extend the number of characters by one to include the "Space character" so that we can have a fixed number of sequences...


# In[3]:


from collections import OrderedDict

enc_len = 50
num_actions = 21

char_pairs = [('A', 0), ('R', 1), ('N', 2), ('D', 3), ('C', 4), ('E', 5), ('Q', 6), ('G', 7), ('H', 8), ('I', 9), ('L', 10), ('K', 11), ('M', 12), ('F', 13), ('P', 14), ('S', 15), ('T', 16), ('W', 17), ('Y', 18), ('V', 19), ('>', 20)]
mol_enc = OrderedDict(char_pairs)
enc_mol = OrderedDict(list(map(lambda x : (x[1], x[0]), char_pairs)))


# In[4]:


def seq_to_enc(seq):
    enc = []
    for i in range(enc_len):
        if i < len(seq):
            enc.append(mol_enc[seq[i]])
            if seq[i] == '>':
                break
        else:
            enc[i].append(20)
    while len(enc) < enc_len:
        enc.append(20)
    
    return F.one_hot(torch.tensor(enc), num_classes=num_actions).numpy()


# In[5]:


def convertor(sequences):
    """
        Does the padding of the sequences to the correct length... w/ the extra chars...
        
        Input: sequences List[str]
        
        Return: list[ndarray]
    """
    
    all_seqs = []
    for seq in sequences:
        all_seqs.append(seq_to_enc(seq).flatten())
        
    return np.stack(all_seqs)
    
    
    


# In[6]:


import pickle

class AMPLandscape(flexs.Landscape):
    """AMP landscape."""

    def __init__(self, mode):
        """Create a AMP landscape."""
        super().__init__(name="AMP")
        assert mode == 'val' or mode == 'test'
        self.alphabet = flexs
        
        if mode == 'val':
            fname = "data/metarlbo_rfc_not-alibicani.pkl"
        elif mode == 'test':
            fname = "data/metarlbo_rfc_alibicani.pkl"
                
        with open(fname, "rb") as f:
            self.model = pickle.load(f)


    def _fitness_function(self, sequences):
        """
            Takes as input a list of strings (w/ alphabet of 20)
            
            
            Returns numpy array of scores
        """
        
        np_seqs = convertor(sequences)
        
        scores = self.model.predict_proba(np_seqs)[:, 1]
        
        return scores
        
        
        
    


# In[7]:


landscape = AMPLandscape("test")
alph_chars = list(mol_enc.keys())[:-1]
alphabet=''.join(alph_chars)


# In[8]:


bsize = 250
nRounds = 12

nRuns = 3


# In[9]:


explorer_scores = {}


# In[10]:


import os

logs_dir = f'./analysis/{exp_name}'

os.listdir(logs_dir)

from datetime import datetime
def get_time():
    return datetime.now().isoformat()


# In[11]:


nModelQueries = 4000


# # Random Explorer

# In[12]:


rand_seq_len = 50


def random_start():
    starting_sequence = "".join([np.random.choice(list(alph_chars)) for _ in range(rand_seq_len)])
    return starting_sequence


starting_sequence = random_start()


# In[13]:


def run_random(nModelQueries):
    scores = []

    for _ in range(nRuns):
        starting_sequence = random_start()
        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        random_explorer = baselines.explorers.Random(
            cnn,
            rounds=nRounds,
            mu=1,
            starting_sequence=starting_sequence,
            sequences_batch_size=bsize,
            model_queries_per_batch=bsize,
            alphabet=alphabet
        )

        random_sequences, metadata = random_explorer.run(landscape)
        fname = "Random_Explorer_" + get_time() + ".csv"
        random_sequences.to_csv(os.path.join(logs_dir, fname))
        random_sequences


# In[56]:


# import json

# with open(os.path.join(logs_dir, "random_baseline_results.json"), "w") as f:
#     json.dump(explorer_scores, f)

# with open(os.path.join(logs_dir, "random_baseline_results.json"), "r") as f:
#     print(json.load(f))


# ## Adalead Explorer

# In[14]:


def run_adalead(nModelQueries):

    scores = []

    for _ in range(nRuns):
        starting_sequence = random_start()
        
        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        adalead_explorer = baselines.explorers.Adalead(
            cnn,
            rounds=nRounds,
            starting_sequence=starting_sequence,
            sequences_batch_size=bsize,
            model_queries_per_batch=nModelQueries,
            alphabet=alphabet
        )

        adalead_sequences, metadata = adalead_explorer.run(landscape)
        fname = "Adalead_Explorer_" + get_time() + ".csv"
        adalead_sequences.to_csv(os.path.join(logs_dir, fname))
        adalead_sequences


# In[15]:


# import json

# with open(os.path.join(logs_dir, "adalead_baseline_results.json"), "w") as f:
#     json.dump(explorer_scores, f)


# ## Genetic Explorer

# In[16]:


def run_genetic(nModelQueries):
    for _ in range(nRuns):
        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        genetic_explorer = baselines.explorers.GeneticAlgorithm(
            cnn,

            population_size=8,
            parent_selection_strategy='wright-fisher', # wright-fisher model decides who gets to 'mate'
            beta=0.01,
            children_proportion=0.2,

            rounds=nRounds,
            starting_sequence=starting_sequence,
            sequences_batch_size=bsize,
            model_queries_per_batch=nModelQueries,
            alphabet=alphabet
        )


        genetic_algo_sequences, metadata = genetic_explorer.run(landscape)
        fname = "Genetic_Explorer_" + get_time() + ".csv"
        genetic_algo_sequences.to_csv(os.path.join(logs_dir, fname))
        genetic_algo_sequences


# ## CMAES Explorer

# In[17]:




def run_cmaes(nModelQueries):
    scores = []

    for _ in range(nRuns):
        starting_sequence = random_start()

        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        cmaes_explorer = baselines.explorers.CMAES(
            flexs.LandscapeAsModel(landscape),

            population_size=10,
            max_iter=200,

            rounds=nRounds,
            starting_sequence=starting_sequence,
            sequences_batch_size=bsize,
            model_queries_per_batch=nModelQueries,
            alphabet=alphabet
        )
        
        cmaes_sequences, metadata = cmaes_explorer.run(landscape)
        fname = "CMAES_Explorer_" + get_time() + ".csv"
        cmaes_sequences.to_csv(os.path.join(logs_dir, fname))
        cmaes_sequences
        
    


# In[18]:


# import json

# with open(os.path.join(logs_dir, "cmaes_baseline_results.json"), "w") as f:
#     json.dump(explorer_scores, f)


# ## DynaPPO Explorer

# In[19]:


def run_dynappo(nModelQueries):
    nModelRounds = nRounds
    scores = []

    for _ in range(nRuns):
        starting_sequence = random_start()
        dynappo_explorer = baselines.explorers.DynaPPO(  # DynaPPO has its own default ensemble model, so don't use CNN
            landscape=landscape,
            env_batch_size=10,
            num_model_rounds=nModelRounds,
            rounds=nRounds,
            starting_sequence=starting_sequence,
            sequences_batch_size=bsize,
            model_queries_per_batch=nModelQueries,
            alphabet=alphabet,
        )

        dynappo_sequences, metadata = dynappo_explorer.run(landscape)
        fname = "DynaPPO_Explorer_" + get_time() + ".csv"
        dynappo_sequences.to_csv(os.path.join(logs_dir, fname))
        dynappo_sequences



def run(args):
    if args.method == 'dynappo':
        run_dynappo(args.nModelQueries)
    elif args.method == 'cmaes':
        run_cames(args.nModelQueries)
    elif args.method == 'genetic':
        run_genetic(args.nModelQueries)
    elif args.method == 'adalead':
        run_adalead(args.nModelQueries)
    elif args.method == 'random':
        run_random(args.nModelQueries)