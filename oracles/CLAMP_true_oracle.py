from oracles.base import BaseOracle
from torch.utils.data import DataLoader
import numpy as np

from data.dynappo_data import enc_to_seq
from tqdm import tqdm


import torch, pickle, gzip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CLAMPTrueOracle(BaseOracle):
    def __init__(self, model_type, mb = 256):
        self.query_count = 0
        self.model_type = model_type
        self.mb = mb

        # GFLowNet Oracle Stuff
        self.tokenizer = pickle.load(gzip.open('data/tokenizer.pkl.gz', 'rb'))
        self.eos_tok = self.tokenizer.numericalize(self.tokenizer.eos_token).item()

    def query(self, model, x, flatten_input=False):
        """
            Args:
             - model: 
             - x: (batch_size, dim of query)
             - flatten_input: True if the model takes in the whole seq. at once (False o.w.)
            
            Return:
             - Reward (Real Number): (batch_size, 1)
        """
        
        batch_size = x.shape[0]
        seqs = []
        for i in range(batch_size):
            seq = enc_to_seq(x[i])

            seq = seq[:seq.find(">")]
            seqs.append(seq)

        print(seqs)
        self.query_count += batch_size

        samples = seqs
        scores = []
        mbsize = 256
        sigmoid = torch.nn.Sigmoid()
        for i in tqdm(range(int(np.ceil(len(samples) / mbsize)))):

            if self.model_type == "GFN":
                x = self.tokenizer.process(samples[i*mbsize:(i+1)*mbsize]).to(device)

                logit = model(x.swapaxes(0,1), x.lt(self.eos_tok)).squeeze(1)
                s = sigmoid(logit)
                scores += s.tolist()
            else:
                s = model.evaluate_many(samples[i*mbsize:(i+1)*mbsize])
                if type(s) == dict:
                    scores += s["confidence"][:, 1].tolist()
                else:
                    scores += s.tolist()
        try:
            return scores
        except:
            return scores



    def fit(self, model, flatten_input=False):
        """
            Fits the model on the entirety of the storage (unneeded since CLAMP oracle is already trained...)

        """

        return model



