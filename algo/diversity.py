import torch
import Bio
from Bio.Blast.Applications import NcbiblastpCommandline
from io import StringIO
from Bio.Blast import NCBIXML
import utils.helpers as utl


def hamming_distance(history, seq):
    """
        Args:
            - history: (length, 50, 21)
            - seq: (50, 21)

        Return:
            - Hamming Distance between seq and each in "history"
               - (length, )
    """
    seq_length = seq.shape[0]
    return ((1 - history) * seq + history * (1 - seq)).sum([1, 2]) / 2 / seq_length # Average Hamming distance (by seq. length) between the strings? 


def pairwise_hamming_distance(history):
    n = history.shape[0]

    ret = 0
    for i in range(n):
        ret += hamming_distance(history, history[i]).sum()
    ret /= n*(n-1)
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


class diversity():
    """ since different distance metric could lead to 'different scale', we should pay attention if
    we want use these metric for comparison purposes. """
    def __init__(self, seq, history, div_switch="ON", radius=2, div_metric_name="hamming"):
        super(diversity, self).__init__()
        self.div_switch=div_switch
        self.seq = seq
        self.history = torch.stack(history)
        self.radius = radius
        # self.history = history
        self.div_metric_name = div_metric_name



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

            # Hamming Distance is equiv. to XOR, but in our case, it's not exactly XOR since we're one hot encoding characters.
            sums = (((1 - self.history) * self.seq + self.history * (1 - self.seq)).sum([1, 2]))/2 

            penalty_sums = sums[(sums < self.radius)]

            # ret = (sums < self.radius).sum().item()

            ret = ((self.radius - penalty_sums)/self.radius).sum().item() # Linear penalty weighting...

            return ret

        else:
            return 0.0

    def density_blast(self):
        if self.div_switch == "ON":
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


    def get_density(self):
        if self.div_metric_name == "blast":
            return self.density_blast()
        elif self.div_metric_name == "hamming":
            return self.density_hamming()
        else:
            raise NotImplementedError

