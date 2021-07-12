import torch


class diversity():
    def __init__(self, seq, history, div=False):
        super(diversity, self).__init__()
        self.div=div
        self.seq = torch.tensor(seq)
        self.history = torch.tensor(history)

    def hamming_distance(self, radius=2):
        if self.div:
            mult = self.history * self.seq

            sum = torch.sum(mult, dim=(1,2))
            count=0
            for i in sum:
                if i >= radius:
                    count+=1
            return count / len(self.history)

        else:
            return 0.0

    def nearest_neighbour(self):
        pass
