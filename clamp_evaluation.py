import os
import torch
from utils import filtering
from data.dynappo_data import enc_to_seq, seq_to_enc
from oracles.CLAMP_true_oracle import CLAMPTrueOracle
from common_evaluation.clamp_common_eval.defaults import get_test_oracle
import numpy as np

class Evaluation:

    def __init__(self, config, metalearner):
        self.config = config
        self.metalearner = metalearner
        self.true_oracle = CLAMPTrueOracle(model_type = self.config["CLAMP"]["evaluation"]["actual_model"])
        self.metalearner_true_oracle_model = metalearner.true_oracle_model
        self.actual_true_oracle_model = get_test_oracle(source="D2_target", model=self.config["CLAMP"]["evaluation"]["actual_model"], feature="AlBert")

    def run(self):
        model_names = ["best_batch_mean_policy", "best_batch_max_policy"]
        save_names = ["best_batch_mean", "best_batch_max"]

        for name, sname in zip(model_names, save_names):
            model_path = os.path.join(self.metalearner.logger.full_output_folder, name + ".pt")
            self.metalearner.policy.load_state_dict(torch.load(model_path))

            sampled_mols, _ = self.metalearner.sample_query_mols({}, 
                                                        num_query_proxies=self.config["CLAMP"]["evaluation"]["num_query_proxies"], 
                                                        num_samples_per_proxy=self.config["CLAMP"]["evaluation"]["num_samples_per_proxy"]) 
            sampled_mols = torch.cat(sampled_mols, dim=0)

            selected_mols, (mols, mols_scores) = self.select_molecules(sampled_mols, self.config["CLAMP"]["evaluation"]["num_mols_select"]) 


            selected_seqs, seqs = self.to_seq(selected_mols), self.to_seq(mols)

            # Save mols and mols_scores...
            seqs_and_scores = list(zip(seqs, mols_scores)) # Metalearner scores...
            seqs_and_scores_save_path = os.path.join(self.metalearner.logger.full_output_folder, "seqs_and_scores_" + sname + ".lst")
            torch.save(seqs_and_scores, seqs_and_scores_save_path)

            print("Seqs + Scores:", seqs_and_scores)


            # Query "Actual" oracle...
            selected_mols_scores = self.true_oracle.query(self.actual_true_oracle_model, selected_mols, flatten_input=True) # Actual scores...

            selected_seqs_and_scores = list(zip(selected_seqs, selected_mols_scores))
            seqs_and_scores_save_path = os.path.join(self.metalearner.logger.full_output_folder, "actual_oracle_selected_seqs_" + sname + ".lst")
            torch.save(selected_seqs_and_scores, seqs_and_scores_save_path)
            
            print("Selected Seqs + Scores:", selected_seqs_and_scores)
        

    def select_molecules(self, mols, n):
        """
            Removes duplicates and selects "n" molecules.

            Returns the selected molecules, (unique mols and their scores)
        """
        # Remove duplicate molecules... in current batch
        mols = np.unique(mols, axis=0)

        # Remove Empty molecule... hardcoding this since I'm tired
        seqs = set(self.to_seq(mols))
        if "" in seqs:
            seqs.remove("")
            seqs = list(seqs)
            mols = self.to_enc(seqs)


        # Filtering and select molecules
        mols = torch.tensor(mols)

        mols_scores = torch.tensor(self.true_oracle.query(self.metalearner_true_oracle_model, mols, flatten_input=True))

        _, sorted_idx = torch.sort(mols_scores, descending = True)

        sorted_mols = mols.clone()[sorted_idx]

        selected_mols = filtering.select(self.config, sorted_mols, n)

        return selected_mols, (mols, mols_scores)




    def to_seq(self, mols):
        """
            Transforms [batch_size, 50, 21] => List of size (batch_size) strings 
        """
        ret = []

        for i in range(len(mols)):
            seq = enc_to_seq(torch.tensor(mols[i]))
            seq = seq[:seq.find(">")]
            ret.append(seq)

        return ret

    def to_enc(self, seqs):

        ret = []

        for i in range(len(seqs)):
            enc = seq_to_enc(seqs[i])
            ret.append(enc)

        return np.stack(ret, 0)


