import pdb

import torch
import Bio
from Bio.Blast.Applications import NcbiblastpCommandline
from io import StringIO
from Bio.Blast import NCBIXML
import utils.helpers as utl
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hamming_distance(history, seq):
    """
        Args:
            - history: (length, 50, 21)
            - seq: (50, 21)

        Return:
            - Hamming Distance between seq and each in "history"
               - (length, )
    """
    return ((1 - history) * seq + history * (1 - seq)).sum([1, 2]) / 2 # Average Hamming distance (by seq. length) between the strings? 

def batch_hamming_distance(history, seqs):
    """
        Args:
            - history: (length, 50, 21)
            - seqs: (batch_size, 50, 21)
        Return:
            - Hamming Distance between each of seqs and each in history
               - (batch_size, length)
    """
    batch_size = seqs.shape[0]

    scores = []
    for i in range(batch_size):
        scores.append(hamming_distance(history, seqs[i]))

    return torch.stack(scores)




def pairwise_hamming_distance(history):
    """
        
        Args:
            - history: (length, 50, 21)
            - seq: (50, 21)

        Return:
            - Hamming Distance (divided by sequence length) between seq and each in "history"
               - (length, )
    """
    n = history.shape[0]
    seq_length = history.shape[1]

    ret = 0
    for i in range(n):
        ret += hamming_distance(history, history[i]).sum()
    ret /= n*(n-1)*seq_length
    return ret

def blast_score(query_fasta, subject_fasta):
    """
    We would use blast Score rather than E-value when the database size is changing.
    Note: Higher blast score = Higher similarity
    Note: in case of error with this function try installing "ncbi-blast+", in compute canada you can load it by 'module load blast'
    """
    scores = []
    cmd = NcbiblastpCommandline(query=query_fasta, subject=subject_fasta, outfmt=5)()[0]
    blast_output = NCBIXML.read(StringIO(cmd))
    for alignment in blast_output.alignments: # Note: when sequences don't have any common hits -> no alignement
        for hsp in alignment.hsps:
            scores.append(hsp.score)
    return scores

def blast_density(scores):
    raise NotImplementedError()


def running_std(prevStd, prevLen, newData):

    var_in_rewards = ((prevStd.pow(2) * prevLen) + (newData.var(0) * newData.shape[0])) / (prevLen + newData.shape[0])
    # total_number_rwd = len(in_rewards) + self.obs_memory.total_number_rwd

    return var_in_rewards.sqrt()


def running_mean(prevMean, prevLen, newData):

    std_in_rewards = ((prevMean * prevLen) + newData.sum(0)) / (prevLen + newData.shape[0])

    return std_in_rewards


class diversity():
    """ since different distance metric could lead to 'different scale', we should pay attention if
    we want use these metric for comparison purposes. """
    def __init__(self, seq, history, model, rand_model, optimizer, int_r_history, config, div_switch="ON", radius=2, div_metric_name="hamming"):
        super(diversity, self).__init__()
        self.div_switch = div_switch
        self.seq = seq.to(device)
        self.history = torch.stack(history).to(device)
        if len(self.seq.shape) == len(self.history.shape):
            self.batch_query = True # Querying in batches
        else:
            self.batch_query = False
        self.model = model
        self.rand_model = rand_model
        self.optimizer = optimizer
        self.int_r_history = int_r_history
        self.radius = radius
        # self.history = history
        self.div_metric_name = div_metric_name
        self.config = config
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)



    # def hamming_distance(self, radius=2):
    #     # For the case of fixed length, one_hot_encoded inputs
    #
    #     if self.div:
    #         mult = self.history * self.seq
    #
    #         sum = torch.sum(mult, dim=(1,2))
    #         count=0
    #         for i in sum:
    #             if i >= radius:
    #                 count+=1
    #         return count / len(self.history)
    #
    #     else:
    #         return 0.0




    def density_hamming(self):
        if self.div_switch == "ON":
            if self.batch_query:
                # Hamming Distance is equiv. to XOR, but in our case, it's XOR/2 since we're one hot encoding characters.
               
                sums = batch_hamming_distance(self.history, self.seq)

                batch_size = sums.shape[0]

                ret = []
                for i in range(batch_size):
                    penalty_sums = sums[i][(sums[i] < self.radius)]
                    ret.append(((self.radius - penalty_sums) / self.radius).sum())
                return torch.stack(ret)
            else:
                # Hamming Distance is equiv. to XOR, but in our case, it's XOR/2 since we're one hot encoding characters.
                sums = hamming_distance(self.history, self.seq)

                penalty_sums = sums[(sums < self.radius)]

                ret = ((self.radius - penalty_sums)/self.radius).sum() # Linear penalty weighting...
            return ret

        else:
            return 0.0

    def sequence_density(self):
        """Get average distance to `seq` out of all observed sequences."""
        if self.div_switch == "ON":
            self.seq_fitness = [1] * len(self.history)
            dens = 0
            dist_radius = 2
            for i in range(len(self.history)):
                dist = int(self.hamming_distance(self.history[i]))
                if dist != 0 and dist <= dist_radius:  # TODO the case when dist = 0
                    dens += self.seq_fitness[i] / dist
        else:
            return 0.0

        return dens

    def density_blast(self):
        if self.div_switch == "ON":
            if self.batch_query:
                raise NotImplementedError
            else:
                convert = utl.convertor()
                s_seq = convert.one_hot_to_AA(self.seq)
                utl.make_fasta(s_seq)
                b_score = blast_score("data/seq.fasta", "data/history.fasta")
                # print("score: ", sum(b_score))

                utl.append_history_fasta()

                return sum(b_score)

        else:
            return 0.0


    def nearest_neighbour(self):
        pass

    def RND_int_r(self):
        """ Returns the intrinsic reward, computed by Random Network Distillation(RND) method.
        """
        # from algo.RND import get_int_r

        # log_int_r = get_int_r(self.seq, input_size, hidden_size, output_size)
        # log_error = []
        # TODO: why in the link multiplies by 0.5? https://github.com/wisnunugroho21/reinforcement_learning_ppo_rnd/blob/bea6e8dd578706232ec390d401b0eec030852ff3/PPO_RND/pytorch/ppo_rnd_pytorch.py#L296
        # TODO: why for normalization only divide by std (and not -mean also)
        if self.config["diversity"]["RND_metric"] == "L2":
            error = torch.mean(torch.square(self.model(self.seq) - self.rand_model(self.seq)) * 0.5, axis=-1)
        if self.config["diversity"]["RND_metric"] == "soft":
            student_logits = self.model(self.seq)
            teacher_logits = self.rand_model(self.seq)
            p = F.softmax(student_logits / self.config["diversity"]["T"], dim=1) #log_softmax
            q = F.softmax(teacher_logits / self.config["diversity"]["T"], dim=1)
            # l_kl = F.kl_div(p, q, size_average=False) * (self.config["diversity"]["T"] ** 2) / student_logits.shape[0]
            error = torch.mean(torch.square(p - q) * 0.5, axis=-1)

        elif self.config["diversity"]["RND_metric"] == "cosine":
            error = (self.cos(self.model(self.seq), self.rand_model(self.seq)) + 1) / 2.0

        else:
            NotImplementedError
            
        # self.int_r_history = {"running_std": 0., "len": 0}
        std_in_rewards = running_std(self.int_r_history["running_std"], self.int_r_history["len"], error).detach()
        mean_in_rewards = running_mean(self.int_r_history["running_mean"], self.int_r_history["len"], error).detach()
        self.int_r_history["running_std"] = std_in_rewards
        self.int_r_history["running_mean"] = mean_in_rewards
        self.int_r_history["len"] += error.shape[0]

        int_rew = error / (std_in_rewards + 1e-8)

        log_int_r = int_rew.detach()

        self.optimizer.zero_grad()
        torch.mean(int_rew).backward()
        self.optimizer.step()
        # import pdb; pdb.set_trace()
        return log_int_r




    def get_density(self):
        if self.div_switch == "ON":
            if self.div_metric_name == "blast":
                return self.density_blast()
            elif self.div_metric_name == "hamming":
                return self.density_hamming()
            elif self.div_metric_name == "fitness_weighted_density":
                return self.sequence_density()
            elif self.div_metric_name == "RND":
                return self.RND_int_r()
            else:
                raise NotImplementedError
        else:
            return torch.tensor(0.0)

