import torch
import Bio
from Bio.Blast.Applications import NcbiblastpCommandline
from io import StringIO
from Bio.Blast import NCBIXML


def hamming_distance(history, seq):
    """
        Args:
            - history: (length, 51, 21)
            - seq: (51, 21)

        Return:
            - Hamming Distance between seq and each in "history"
               - (length, )
    """
    return ((1 - history) * seq + history * (1 - seq)).sum([1, 2])


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
    Note: in case of error with this function try installing "ncbi-blast+"
    """
    scores = []
    cmd = NcbiblastpCommandline(query=query_fasta, subject=subject_fasta, outfmt=5)()[0]
    blast_output = NCBIXML.read(StringIO(cmd))
    for alignment in blast_output.alignments:
        for hsp in alignment.hsps:
            scores.append(hsp.score)
    return scores

def blast_density(scores):
    raise NotImplementedError()


class diversity():
    def __init__(self, seq, history, div=False, radius = 2):
        super(diversity, self).__init__()
        self.div=div
        self.seq = seq
        self.history = torch.stack(history)
        self.radius = radius


    def density(self):
        if self.div:
            
            # Hamming Distance is equiv. to XOR
            sums = ((1 - self.history) * self.seq + self.history * (1 - self.seq)).sum([1, 2])

            ret = (sums <= self.radius).sum().item()
            
            return ret

        else:
            return 0.0


    def nearest_neighbour(self):
        pass



