import torch


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
            
            # Hamming Distance is equiv. to XOR
            sums = ((1 - self.history) * self.seq + self.history * (1 - self.seq)).sum([1, 2])

            ret = (sums <= self.radius).sum().item() / len(self.history)  
            
            return ret

        else:
            return 0.0

    def nearest_neighbour(self):
        pass



