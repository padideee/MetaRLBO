#!/usr/bin/env python
# coding: utf-8

# In[33]:


import flexs


# In[34]:


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


# In[35]:


seq_len = 50 # 20 or 50!
exp_name = 'Ising50'


# In[36]:



from oracles.custom_models.alt_ising_model import AlternatingChainIsingModel

def AltIsingModel(length=50, vocab_size=20):
    return AlternatingChainIsingModel(length=length, vocab_size=vocab_size)


model = AltIsingModel(length=seq_len, vocab_size=20)


# In[37]:


from collections import OrderedDict

# enc_len = 50
num_actions = 20


char_pairs = [('A', 0), ('R', 1), ('N', 2), ('D', 3), ('C', 4), ('E', 5), ('Q', 6), ('G', 7), ('H', 8), ('I', 9), ('L', 10), ('K', 11), ('M', 12), ('F', 13), ('P', 14), ('S', 15), ('T', 16), ('W', 17), ('Y', 18), ('V', 19), ('>', 20)]
mol_enc = OrderedDict(char_pairs)
enc_mol = OrderedDict(list(map(lambda x : (x[1], x[0]), char_pairs)))


# In[38]:


def seq_to_enc(seq):
    enc = [None for i in range(len(seq))]
    for i in range(len(seq)):
        enc[i] = mol_enc[seq[i]]
    
    return F.one_hot(torch.tensor(enc), num_classes=num_actions).numpy()


# In[39]:


def convertor(sequences):
    """
        Does the padding of the sequences to the correct length... w/ the extra chars...
        
        Input: sequences List[str]
        
        Return: list[ndarray]
    """
    
    all_seqs = []
    for seq in sequences:
        all_seqs.append(seq_to_enc(seq)) # Not flattened for this problem
        
    return np.stack(all_seqs)
    
    
    


# In[40]:


import pickle

class IsingLandscape(flexs.Landscape):
    """AMP landscape."""

    def __init__(self, seq_len):
        """Create a AMP landscape."""
        super().__init__(name=f"Ising{seq_len}")
        self.alphabet = flexs
        
        self.model = AltIsingModel(length=seq_len, vocab_size=20)


    def _fitness_function(self, sequences):
        """
            Takes as input a list of strings (w/ alphabet of 20)
            
            
            Returns numpy array of scores
        """
        
        np_seqs = convertor(sequences)
        scores = self.model(np_seqs.argmax(-1))
        
        return scores


# In[41]:


def get_scores(sequences, nRounds):
    run_max_scores = []
    for i in range(nRounds):
        max_found = sequences[sequences['round'] <= i+1].true_score.max()
        run_max_scores.append(max_found)
    return run_max_scores
    


# In[42]:


landscape = IsingLandscape(seq_len)
alph_chars = list(mol_enc.keys())[:-1]
alphabet=''.join(alph_chars)


# In[43]:


from datetime import datetime
import os

logs_dir = f'./analysis/{exp_name}'

def get_time():
    return datetime.now().isoformat()

os.listdir(logs_dir)


# In[44]:


query_batch_size = 500
model_queries_per_batch = 4000
nRounds = 16
nRuns = 3


# In[45]:


# Start from a random sequence!

rand_seq_len = seq_len


def random_start():
    starting_sequence = "".join([np.random.choice(list(alph_chars)) for _ in range(rand_seq_len)])
    return starting_sequence

def store_results(results, baseline_name):
    import os
    with open(os.path.join(logs_dir, f"{baseline_name}_baseline_results.json"), "w") as f:
        json.dump(results, f)

starting_sequence = random_start()


# ## Random Explorer

# In[32]:



def run_random(nModelQueries):
    for _ in range(nRuns):
        starting_sequence = random_start()
        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        random_explorer = baselines.explorers.Random(
            cnn,
            rounds=nRounds,
            mu=1,
            starting_sequence=starting_sequence,
            sequences_batch_size=query_batch_size,
            model_queries_per_batch=nModelQueries,
            alphabet=alphabet
        )


        random_sequences, metadata = random_explorer.run(landscape)
        fname = "Random_Explorer_" + get_time() + ".csv"
        random_sequences.to_csv(os.path.join(logs_dir, fname))
        random_sequences


# ## Adalead Explorer

# In[46]:


def run_adalead(nModelQueries):

    for _ in range(nRuns):
        starting_sequence = random_start()
        
        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        adalead_explorer = baselines.explorers.Adalead(
            cnn,
            rounds=nRounds,
            starting_sequence=starting_sequence,
            sequences_batch_size=query_batch_size,
            model_queries_per_batch=nModelQueries,
            alphabet=alphabet
        )



        adalead_sequences, metadata = adalead_explorer.run(landscape)
        fname = "Adalead_Explorer_" + get_time() + ".csv"
        adalead_sequences.to_csv(os.path.join(logs_dir, fname))
        adalead_sequences


# In[47]:


# import json

# with open(os.path.join(logs_dir, "adalead_baseline_results.json"), "w") as f:
#     json.dump(explorer_scores, f)


# ## Genetic Explorer

# In[49]:


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
            sequences_batch_size=query_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            alphabet=alphabet
        )


        genetic_algo_sequences, metadata = genetic_explorer.run(landscape)
        fname = "Genetic_Explorer_" + get_time() + ".csv"
        genetic_algo_sequences.to_csv(os.path.join(logs_dir, fname))
        genetic_algo_sequences


# ## CMAES

# In[48]:


def run_cmaes(nModelQueries):
    for _ in range(nRuns):

        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        cmaes_explorer = baselines.explorers.CMAES(
            flexs.LandscapeAsModel(landscape),

            population_size=10,
            max_iter=200,

            rounds=nRounds,
            starting_sequence=starting_sequence,
            sequences_batch_size=query_batch_size,
            model_queries_per_batch=model_queries_per_batch,
            alphabet=alphabet
        )

        cmaes_sequences, metadata = cmaes_explorer.run(landscape)
        fname = "CMAES_Explorer_" + get_time() + ".csv"
        cmaes_sequences.to_csv(os.path.join(logs_dir, fname))
        cmaes_sequences


# ## DynaPPO

# In[45]:



def run_dynappo(nModelQueries):
    scores = []
    nModelRounds = nRounds

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


# In[ ]:


# import json

# with open(os.path.join(logs_dir, "dynappo_baseline_results.json"), "w") as f:
#     json.dump(explorer_scores, f)


# ## PPO

# In[50]:

def run_ppo(nModelQueries):

    for _ in range(nRuns):
        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        ppo_explorer = baselines.explorers.PPO(  # DynaPPO has its own default ensemble model, so don't use CNN
            model=cnn,
            rounds=nRounds,
            starting_sequence=starting_sequence,
            sequences_batch_size=query_batch_size,
            model_queries_per_batch=nModelQueries,
            alphabet=alphabet,
        )
        ppo_sequences, metadata = ppo_explorer.run(landscape)
        fname = "PPO_Explorer_" + get_time() + ".csv"
        ppo_sequences.to_csv(os.path.join(logs_dir, fname))
        ppo_sequences




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
    elif args.method == 'ppo':
        run_ppo(args.nModelQueries)
        