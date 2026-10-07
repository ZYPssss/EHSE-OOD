import argparse

import numpy as np
import os.path as osp
import torch
import torch.utils
import torch.utils.data
from torch_geometric.data import InMemoryDataset, Data
import mmcv
import rdkit
from GOOD import register
from torch_geometric.loader import DataLoader

def GetDataLoader(args, d_name, domain, shift):
    device = torch.device('cuda', args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    args.device = device


    dataset, meta_info = register.datasets[d_name].load(dataset_root=osp.join(args.data_root, args.dataset),
                                                               domain=domain,
                                                               shift=shift,
                                                               generate=False,)


    iid_dataset = dataset["train"]
    # ood_dataset = DrugOOD(osp.join(dataset_dir, 'DrugOOD/'), mode='ood')

    n_train_data, n_val_data, n_in_test_data, n_out_test_data = 1000, 500, 500, 500
    train_dataset, in_test_dataset = iid_dataset[:n_train_data], dataset["id_test"][:n_in_test_data]

    out_test_dataset = dataset["test"][:n_out_test_data]

    # we need to modify outliers' idx to track their gradients (for GraphDE-v)

    id_test_loader = DataLoader(in_test_dataset, batch_size=args.batch_size, shuffle=False)
    ood_test_loader = DataLoader(out_test_dataset, batch_size=args.batch_size, shuffle=False)

    return id_test_loader, ood_test_loader



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data-Centric Learning from Unlabeled Graphs with Diffusion Model')
    parser.add_argument('--gpu-id', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='number of workers for data loader')
    parser.add_argument('--no-print', action='store_true', default=False,
                        help="don't use progress bar")

    parser.add_argument('--dataset', default="GOOD/", type=str,
                        choices=['DrugOOD/', 'GOOD/'],
                        help='dataset name (plym-, ogbg-)')

    parser.add_argument('--data-root', default="./data/", type=str)

    # training
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stop')
    parser.add_argument('--trails', type=int, default=5,
                        help='nubmer of experiments (default: 5)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-2,
                        help='Learning rate (default: 1e-2)')
    parser.add_argument('--wdecay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--initw-name', type=str, default='default',
                        help="method to initialize the model paramter")
    # augmentation
    parser.add_argument('--start', type=int, default=20,
                        help="start epoch for augmentation")
    parser.add_argument('--iteration', type=int, default=20,
                        help='epoch to do augmentation')
    parser.add_argument('--strategy', default="replace_accumulate", type=str,
                        choices=['replace_once', 'add_once', 'replace_accumulate', 'add_accumulate'],
                        help='  strategy about how to use the augmented examples. \
                                    Replace or add to the original examples; Accumulate the augmented examples or not')
    parser.add_argument('--n-jobs', type=int, default=22,
                        help='# process to convert the dense adj input to pyg input form')
    parser.add_argument('--n-negative', type=int, default=5,
                        help='# negative samples to optimize the augmented example')
    parser.add_argument('--out-steps', type=int, default=5,
                        help='outer sampling steps for guided reverse diffusion')
    parser.add_argument('--topk', type=int, default=10,
                        help='top k in an augmentation batch ')
    parser.add_argument('--aug-batch', type=int, default=2000,
                        help='the  augmentation batch compared to training batch')
    parser.add_argument('--snr', type=float, default=0.2,
                        help='snr')
    parser.add_argument('--scale-eps', type=float, default=0,
                        help='scale eps')
    parser.add_argument('--perturb-ratio', type=float, default=None,
                        help='level of noise for perturbation')
    args = parser.parse_args()

    args.strategy_init = args.strategy

    dataset_name = ['GOODHIV_scaffold_concept', 'GOODHIV_scaffold_covariate',
                    'GOODHIV_size_concept', 'GOODHIV_size_covariate',
                    'GOODPCBA_scaffold_concept', 'GOODPCBA_scaffold_covariate',
                    'GOODPCBA_size_concept', 'GOODPCBA_size_covariate',
                    'GOODZINC_scaffold_concept', 'GOODZINC_scaffold_covariate',
                    'GOODZINC_size_concept', 'GOODZINC_size_covariate']


    for i in range(len(dataset_name)):
        d_name, domain, shift = dataset_name[i].split('_')
        GetDataLoader(args, d_name, domain, shift)

