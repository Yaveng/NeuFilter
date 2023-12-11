from train import train
from eval_test import eval_epoch
import numpy as np
import random
import os
import torch
import sys
import argparse


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    # torch.backends.cudnn.deterministic = True  # We find this code will cause some issues, thus we disable it.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeuFilter')
    parser.add_argument('--network', default='wikipedia', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--num_layer', default=2, type=int)
    parser.add_argument('--reg_factor1', default=0.001, type=float)
    parser.add_argument('--reg_factor2', default=0.001, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--train_proportion', default=0.6, type=float)
    parser.add_argument('--add_state_change_loss', default=True)
    args = parser.parse_args()
    args.postfix = '{}-{}-{}-{}-{}-{}'.format(args.network, args.embedding_dim, args.num_layer, args.reg_factor1, args.reg_factor2, args.lr)

    set_seed(args.seed)

    print('Start training {} ...'.format(args.network))
    train(args)
    print('Training {} successfully'.format(args.network))
    print('Start evaluating {} ...'.format(args.network))
    valid_results = []
    for ep in range(args.epochs):
        args.epoch = ep
        valid_result = eval_epoch(args, mode='valid')
        valid_results.append(valid_result)
    valid_results = np.array(valid_results)
    best_val_idx = np.argmax(valid_results[:, 0])
    args.epoch = int(valid_results[best_val_idx, 1])
    eval_epoch(args, mode='test')
    print('Evaluating {} successfully'.format(args.network))