import torch

def hamming_distance(history, seq):
    """
        Args:
            - history: (length, 50, 21)
            - seq: (50, 21)

        Return:
            - Hamming Distance between seq and each in "history"
               - (length, )
    """
    seq_length = seq.shape[1]
    return ((1 - history) * seq + history * (1 - seq)).sum([1, 2]) / seq_length


def pairwise_hamming_distance(history):
    n = history.shape[0]

    ret = 0
    for i in range(n):
        ret += hamming_distance(history, history[i]).sum()
    ret /= n*(n-1)
    return ret




class diversity():
    def __init__(self, seq, history, div=False, radius = 2):
        super(diversity, self).__init__()
        self.div=div
        self.seq = seq
        self.history = torch.stack(history)
        self.radius = radius

    # def hamming_distance(self, radius=2):
    #     if self.div:
    #         mult = self.history * self.seq

    #         sum = torch.sum(mult, dim=(1,2))
    #         count=0
    #         for i in sum:
    #             if i >= radius:
    #                 count+=1
    #         return count / len(self.history)

    #     else:
    #         return 0.0


    def density(self):
        if self.div:
            
            # Hamming Distance is equiv. to XOR, but in our case, it's not exactly XOR since we're one hot encoding characters.
            sums = (((1 - self.history) * self.seq + self.history * (1 - self.seq)).sum([1, 2]))/2

            penalty_sums = sums[(sums < self.radius)]
            
            # ret = (sums < self.radius).sum().item() 

            ret = ((self.radius - penalty_sums)/2).sum().item() # Linear penalty weighting...

            return ret

        else:
            return 0.0

    def nearest_neighbour(self):
        pass



