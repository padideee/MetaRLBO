#!/usr/bin/env python
# coding: utf-8

# In[2]:


import flexs


# In[3]:


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


# In[4]:


exp_name = 'RNA14'


# In[5]:




problem = flexs.landscapes.rna.registry()['L14_RNA1']
pprint.pprint(problem)

landscape = flexs.landscapes.RNABinding(**problem['params'])
alphabet = s_utils.RNAA


bsize=100


# In[6]:


nModelQueries = 2000
nRounds = 15


# In[7]:


seq_length = problem['params']['seq_length']


# In[8]:


import random

starting_sequence = "".join([random.choice(alphabet) for i in range(seq_length)])


# In[9]:


import os

logs_dir = f'./analysis/{exp_name}'

os.listdir(logs_dir)


from datetime import datetime
def get_time():
    return datetime.now().isoformat()


# ## Random Explorer

# In[34]:

def run_random(nModelQueries):
    for _ in range(3):
        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        random_explorer = baselines.explorers.Random(
            cnn,
            rounds=nRounds,
            mu=1,
            starting_sequence=starting_sequence,
            sequences_batch_size=100,
            model_queries_per_batch=nModelQueries,
            alphabet=alphabet
        )

        random_sequences, metadata = random_explorer.run(landscape)
        fname = "Random_Explorer_" + get_time() + ".csv"
        random_sequences.to_csv(os.path.join(logs_dir, fname))
        random_sequences


# ## Adalead Explorer

# In[35]:



def run_adalead(nModelQueries):
    for _ in range(3):

        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        adalead_explorer = baselines.explorers.Adalead(
            cnn,
            rounds=nRounds,
            starting_sequence=starting_sequence,
            sequences_batch_size=100,
            model_queries_per_batch=nModelQueries,
            alphabet=alphabet
        )


        adalead_sequences, metadata = adalead_explorer.run(landscape)
        fname = "Adalead_Explorer_" + get_time() + ".csv"
        adalead_sequences.to_csv(os.path.join(logs_dir, fname))
        adalead_sequences


# ## Genetic Explorer

# In[10]:

def run_genetic(nModelQueries):
    for _ in range(3):
        
        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        genetic_explorer = baselines.explorers.GeneticAlgorithm(
            cnn,

            population_size=20,
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

# In[11]:

def run_cmaes(nModelQueries):
    for _ in range(3):
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


# ## DynaPPO Explorer

# In[12]:
def run_dynappo(nModelQueries):

    for _ in range(3):

        cnn = baselines.models.CNN(len(starting_sequence), alphabet=alphabet,
                                 num_filters=32, hidden_size=100, loss='MSE')

        dynappo_explorer = baselines.explorers.DynaPPO(  # DynaPPO has its own default ensemble model, so don't use CNN
            landscape=landscape,
            env_batch_size=10,
            num_model_rounds=10,
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
        