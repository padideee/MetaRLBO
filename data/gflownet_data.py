
class ALDataset:
    def __init__(self, split, nfold, args, oracle):
        source = get_default_data_splits(setting='Target')
        self.rng = np.random.RandomState(142857)
        self.data = source.sample(split, -1)
        self.nfold = nfold
        if split == "D1": groups = np.array(source.d1_pos.group)
        if split == "D2": groups = np.array(source.d2_pos.group)
        if split == "D": groups = np.concatenate((np.array(source.d1_pos.group), np.array(source.d2_pos.group)))

        n_pos, n_neg = len(self.data['AMP']), len(self.data['nonAMP'])
        pos_train, pos_valid = next(GroupKFold(nfold).split(np.arange(n_pos), groups=groups))
        neg_train, neg_valid = next(GroupKFold(nfold).split(np.arange(n_neg),
                                                            groups=self.rng.randint(0, nfold, n_neg)))
        self.pos_train = [self.data['AMP'][i] for i in pos_train]
        self.neg_train = [self.data['nonAMP'][i] for i in neg_train]
        self.pos_valid = [self.data['AMP'][i] for i in pos_valid]
        self.neg_valid = [self.data['nonAMP'][i] for i in neg_valid]

        print("Getting scores")
        if osp.exists("cls" + split+"pos_train_scores.npy"):
            self.pos_train_scores = np.load("cls" + split+"pos_train_scores.npy")
        else:
            self.pos_train_scores, _ = query_oracle(args, oracle, self.pos_train, None)
            np.save("cls" + split+"pos_train_scores.npy", self.pos_train_scores)
        
        if osp.exists("cls" + split+"neg_train_scores.npy"):
            self.neg_train_scores = np.load("cls" + split+"neg_train_scores.npy")
        else:
            self.neg_train_scores, _ = query_oracle(args, oracle, self.neg_train, None)
            np.save("cls" + split+"neg_train_scores.npy", self.neg_train_scores)
        
        if osp.exists("cls" + split+"pos_val_scores.npy"):
            self.pos_valid_scores = np.load("cls" + split+"pos_val_scores.npy")
        else: 
            self.pos_valid_scores, _ = query_oracle(args, oracle, self.pos_valid, None)
            np.save("cls" + split+"pos_val_scores.npy", self.pos_valid_scores)
        
        if osp.exists("cls" + split+"neg_val_scores.npy"):
            self.neg_valid_scores = np.load("cls" + split+"neg_val_scores.npy")
        else:
            self.neg_valid_scores, _ = query_oracle(args, oracle, self.neg_valid, None)
            np.save("cls" + split+"neg_val_scores.npy", self.neg_valid_scores)

    def sample(self, n, r):
        n_pos = int(np.ceil(n * r))
        n_neg = n - n_pos
        return (([self.pos_train[i] for i in np.random.randint(0, len(self.pos_train), n_pos)] +
                 [self.neg_train[i] for i in np.random.randint(0, len(self.neg_train), n_neg)]),
                [1] * n_pos + [0] * n_neg)

    def get_valid(self):
        return self.pos_valid + self.neg_valid, [1] * len(self.pos_valid) + [0] * len(self.neg_valid)

    def add(self, samples, oracle_preds):
        scores, labels = oracle_preds
        neg_val, neg_train, pos_train, pos_val = [], [], [], []
        for x, y, score in zip(samples, labels, scores):
            if np.random.uniform() < (1/self.nfold):
                if y == 0:
                    self.neg_valid.append(x)
                    neg_val.append(score)
                else:
                    self.pos_valid.append(x)
                    pos_val.append(score)
            else:
                if y == 0:
                    self.neg_train.append(x)
                    neg_train.append(score)
                else:
                    self.pos_train.append(x)
                    pos_train.append(score)
        self.neg_train_scores = np.concatenate((self.neg_train_scores, neg_train), axis=0)
        self.pos_train_scores = np.concatenate((self.pos_train_scores, pos_train), axis=0)
        self.neg_valid_scores = np.concatenate((self.neg_valid_scores, neg_val), axis=0)
        self.pos_valid_scores = np.concatenate((self.pos_valid_scores, pos_val), axis=0)
