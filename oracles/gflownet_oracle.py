import argparse
import gzip
import pickle
import multiprocessing as mp
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from oracles.custom_models.gfn_transformer import GFNTransformer
# from utils.distance import is_similar, diamond_dist, blast_dist, edit_dist
from common_evaluation.clamp_common_eval.defaults import get_test_oracle, get_default_data_splits

parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/al_0.pkl.gz')

# Multi-round
parser.add_argument("--num_rounds", default=5, type=int)
parser.add_argument("--num_sampled_per_round", default=10000)
parser.add_argument("--num_folds", default=5)
parser.add_argument("--round_continuation", default="finetune")
parser.add_argument("--seed", default=0)
parser.add_argument("--noise_params", action="store_true")
parser.add_argument("--save_proxy_weights", action="store_true")
parser.add_argument("--use_uncertainty", action="store_true")
parser.add_argument("--clearml", action="store_true")
parser.add_argument("--filter", action="store_true")
parser.add_argument("--proxy_kappa", default=0.1, type=float)
parser.add_argument("--acq_fn", default="UCB", type=str)
parser.add_argument("--load_proxy_weights", type=str)
parser.add_argument("--max_percentile", default=90, type=int)
parser.add_argument("--filter_threshold", default=0.1, type=float)
parser.add_argument("--filter_distance_type", default="edit", type=str)
parser.add_argument("--oracle_split", default="D2_target", type=str)
parser.add_argument("--proxy_data_split", default="D1", type=str)
parser.add_argument("--oracle_type", default="MLP", type=str)
parser.add_argument("--oracle_features", default="AlBert", type=str)
parser.add_argument("--medoid_oracle_dist", default="diamond", type=str)
parser.add_argument("--medoid_oracle_norm", default=35, type=int)
parser.add_argument("--medoid_oracle_exp_constant", default=6, type=int)


# Generator
parser.add_argument("--gen_learning_rate", default=5e-5, type=float)
parser.add_argument("--gen_num_iterations", default=20000, type=int) # Maybe this is too low?
parser.add_argument("--gen_episodes_per_step", default=16, type=int)
parser.add_argument("--gen_num_hid", default=128, type=int)
parser.add_argument("--gen_reward_norm", default=1, type=float)
parser.add_argument("--gen_reward_exp", default=8, type=float)
parser.add_argument("--gen_reward_min", default=-8, type=float)
# Soft-QLearning/GFlownet gen
parser.add_argument("--gen_reward_exp_ramping", default=1, type=float)
parser.add_argument("--gen_balanced_loss", default=1, type=float)
parser.add_argument("--gen_output_coef", default=10, type=float)
parser.add_argument("--gen_loss_eps", default=1e-5, type=float)
parser.add_argument("--gen_random_action_prob", default=0.0001, type=float)
parser.add_argument("--gen_sampling_temperature", default=1., type=float)
parser.add_argument("--gen_leaf_coef", default=25, type=float)
parser.add_argument("--gen_data_sample_per_step", default=16, type=int)
# PG gen
parser.add_argument("--gen_do_pg", default=1, type=int)
parser.add_argument("--gen_pg_entropy_coef", default=1e-1, type=float)

# Proxy
parser.add_argument("--proxy_learning_rate", default=1e-4)
parser.add_argument("--proxy_type", default="classifier")

parser.add_argument("--proxy_num_hid", default=64, type=int)
parser.add_argument("--proxy_L2", default=1e-4, type=float)
parser.add_argument("--proxy_num_per_minibatch", default=256, type=int)
parser.add_argument("--proxy_early_stop_tol", default=5, type=int)
parser.add_argument("--proxy_num_iterations", default=30000, type=int)
parser.add_argument("--proxy_num_droput_sample", default=50, type=int)
parser.add_argument("--proxy_pos_ratio", default=0.9, type=float)


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

        # print("Getting scores")
        # if osp.exists("cls" + split+"pos_train_scores.npy"):
        #     self.pos_train_scores = np.load("cls" + split+"pos_train_scores.npy")
        # else:
        #     self.pos_train_scores, _ = query_oracle(args, oracle, self.pos_train, None)
        #     np.save("cls" + split+"pos_train_scores.npy", self.pos_train_scores)
        
        # if osp.exists("cls" + split+"neg_train_scores.npy"):
        #     self.neg_train_scores = np.load("cls" + split+"neg_train_scores.npy")
        # else:
        #     self.neg_train_scores, _ = query_oracle(args, oracle, self.neg_train, None)
        #     np.save("cls" + split+"neg_train_scores.npy", self.neg_train_scores)
        
        # if osp.exists("cls" + split+"pos_val_scores.npy"):
        #     self.pos_valid_scores = np.load("cls" + split+"pos_val_scores.npy")
        # else: 
        #     self.pos_valid_scores, _ = query_oracle(args, oracle, self.pos_valid, None)
        #     np.save("cls" + split+"pos_val_scores.npy", self.pos_valid_scores)
        
        # if osp.exists("cls" + split+"neg_val_scores.npy"):
        #     self.neg_valid_scores = np.load("cls" + split+"neg_val_scores.npy")
        # else:
        #     self.neg_valid_scores, _ = query_oracle(args, oracle, self.neg_valid, None)
        #     np.save("cls" + split+"neg_val_scores.npy", self.neg_valid_scores)

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

def query_oracle(args, oracle, samples, outer_loop_iter, mbsize=256):
    print("Computing oracle scores")

    scores = []
    labels = []
    for i in tqdm(range(int(np.ceil(len(samples) / mbsize)))):
        s = oracle.evaluate_many(samples[i*mbsize:(i+1)*mbsize])
        if type(s) == dict:
            scores += s["confidence"][:, 1].tolist()
            labels += s["prediction"].tolist()
        else:
            scores += s.tolist()
            labels += (s > 0.5).astype(int).tolist()

        
    return np.float32(scores), labels


def train_proxy(args, proxy, data, outer_loop_iter):
    print("Training proxy")
    model, opt = proxy

    tokenizer = pickle.load(gzip.open('data/tokenizer.pkl.gz', 'rb'))
    eos_tok = tokenizer.numericalize(tokenizer.eos_token).item()
    sigmoid = torch.nn.Sigmoid()
    device = args.device

    losses = []
    accs = []
    test_losses = []
    test_accs = []
    best_params = None
    best_accuracy = 0
    best_loss = 1e6
    early_stop_tol = args.proxy_early_stop_tol
    early_stop_count = 0
    # There's about 3.2k examples, with mbsize 256 that's 12
    # minibatches per epoch... seems low?
    epoch_length = 100

    import pdb; pdb.set_trace()

    for it in tqdm(range(args.proxy_num_iterations)):
        x, y = data.sample(args.proxy_num_per_minibatch, args.proxy_pos_ratio)
        x = tokenizer.process(x).to(device)
        y = torch.tensor(y, device=device, dtype=torch.float)
        logit = model(x.swapaxes(0,1), x.lt(eos_tok)).squeeze(1)
        if args.proxy_type == "classifier":
            s = sigmoid(logit)
            nll = -torch.log(s) * y - torch.log(1-s) * (1-y)
            loss = nll.mean()
        elif args.proxy_type == "regressor":
            mse = (logit - y).pow(2)
            loss = mse.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        if args.proxy_type == "classifier":
            acc = ((s > 0.5) == y).float().mean()
            accs.append(acc.item())
        
        losses.append(loss.item())

        if not it % epoch_length:
            vx, vy = data.get_valid()
            accuracies = []
            vlosses = []
            for j in range(len(vx) // 256):
                x = tokenizer.process(vx[j*256:(j+1)*256]).to(device)
                y = torch.tensor(vy[j*256:(j+1)*256], device=device, dtype=torch.float)
                logit = model(x.swapaxes(0,1), x.lt(eos_tok)).squeeze(1)
                if args.proxy_type == "classifier":
                    s = sigmoid(logit)
                    nll = -torch.log(s)
                    loss = -torch.log(s) * y - torch.log(1-s) * (1-y)
                    accuracies.append(((s > 0.5) == y).float().sum().item())
                elif args.proxy_type == "regressor":
                    loss = (logit - y).pow(2)
                vlosses.append(loss.sum().item())
            
            if args.proxy_type == "classifier":
                accuracy = np.sum(accuracies) / len(vx)
            test_loss = np.sum(vlosses) / len(vx)
            test_losses.append(test_loss)
            
            if args.proxy_type == "classifier":
                test_accs.append(accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = [i.data.cpu().numpy() for i in model.parameters()]
                    early_stop_count = 0
                else:
                    early_stop_count += 1
            elif args.proxy_type == "regressor":
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_params = [i.data.cpu().numpy() for i in model.parameters()]
                    early_stop_count = 0
                else:
                    early_stop_count += 1

            if early_stop_count >= early_stop_tol:
                print(best_accuracy)
                print('early stopping')
                break
    # Put best parameters back in
    for i, besti in zip(model.parameters(), best_params):
        i.data = torch.tensor(besti).to(device)

    if args.save_proxy_weights:
        torch.save(model.state_dict(), osp.join("".join(args.save_path.split('/')[:-1]), "proxy_{}.pt".format(outer_loop_iter)))

    args.logger.add('proxy_losses', losses)
    args.logger.add('proxy_test_losses', test_losses)

    if args.proxy_type == "classifier":
        args.logger.add('proxy_accs', accs)
        args.logger.add('proxy_test_accs', test_accs)


def make_proxy(args):
    num_tokens = 23
    max_len = 120

    model = GFNTransformer(num_tokens=num_tokens,
                           num_outputs=1,
                           num_hid=args.proxy_num_hid,
                           num_layers=4, # TODO: add these as hyperparameters?
                           num_head=8,
                           max_len=max_len)

    model.to(args.device)
    opt = torch.optim.Adam(model.parameters(), args.proxy_learning_rate,
                           weight_decay=args.proxy_L2)
    return [model, opt]


def main(args):

    # args.device = device = torch.device('cuda')
    args.device = device = torch.device('cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # dist_fn = None
    # if args.medoid_oracle_dist == "edit":
    #     dist_fn = edit_dist
    # elif args.medoid_oracle_dist == "diamond":
    #     dist_fn = diamond_dist
    # elif args.medoid_oracle_dist == "blast":
    #     dist_fn = blast_dist
    # TODO: add title/target argument from args
    oracle = get_test_oracle(args.oracle_split, model=args.oracle_type, feature=args.oracle_features)

    # TODO: add title/target argument from args
    proxy_dataset = ALDataset(args.proxy_data_split, args.num_folds, args, oracle) 

    proxy = make_proxy(args)

    if args.load_proxy_weights is not None:
        try:
            proxy[0].load_state_dict(args.load_proxy_weights)
        except:
            print('Could not load proxy weights, starting from scratch')
            train_proxy(args, proxy, proxy_dataset, 0)
    else:
        train_proxy(args, proxy, proxy_dataset, 0)
        
    args.logger = None
    return proxy


# def run():
if __name__ == '__main__':
    args = parser.parse_args()
    print(args.save_path.split('/')[-1].split('.')[0])
    main(args)