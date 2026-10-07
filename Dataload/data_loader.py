import os
import re
import os.path as osp
from scipy import sparse as sp
import torch
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx
from ogb.graphproppred import PygGraphPropPredDataset

def init_structural_encoding(gs, rw_dim=16, dg_dim=16):
    for g in gs:
        A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
        D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

        Dinv = sp.diags(D)
        RW = A * Dinv
        M = RW

        RWSE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(rw_dim-1):
            M_power = M_power * M
            RWSE.append(torch.from_numpy(M_power.diagonal()).float())
        RWSE = torch.stack(RWSE,dim=-1)

        g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(0, dg_dim - 1)
        DGSE = torch.zeros([g.num_nodes, dg_dim])
        for i in range(len(g_dg)):
            DGSE[i, int(g_dg[i])] = 1

        g['x_s'] = torch.cat([RWSE, DGSE], dim=1)

    return gs



def get_ood_dataset(args, train_per=0.9, need_str_enc=True, pre=False):
    if args.DS_pair is not None:
        DSS = args.DS_pair.split("+")
        DS, DS_ood = DSS[0], DSS[1]
    else:
        DS, DS_ood = args.DS, args.DS_ood

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    path_ood = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS_ood)

    if pre:
        from Dataload.aug import PygGraphPropPredDataset_aug
        from Dataload.codingTree import GraphTransform
        degreeDim = 10
        # degreeDim = 10 if args.DS in ['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K'] else 100
        pre_transform = GraphTransform(args.tree_depth, degreeDim).transform
        processed_filename = 'data_tree%s.pt' % args.tree_depth

        import time
        start_time = time.time()

        dataset = PygGraphPropPredDataset_aug(name=DS, root=path, pre_transform=pre_transform,
                                              processed_filename=processed_filename)
        dataset.data.x = dataset.data.x.type(torch.float32)
        dataset_ood = PygGraphPropPredDataset_aug(name=DS_ood, root=path_ood, pre_transform=pre_transform,
                                                  processed_filename=processed_filename)
        dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)

        end_time = time.time()
        runtime = end_time - start_time
        print(f"Coding tree construction time: {runtime:.2f} s")

    else:
        dataset = PygGraphPropPredDataset(name=DS, root=path)
        dataset.data.x = dataset.data.x.type(torch.float32)
        dataset_ood = PygGraphPropPredDataset(name=DS_ood, root=path_ood)
        dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)

    dataset_num_features = dataset.num_node_features
    dataset_num_features_ood = dataset_ood.num_node_features
    assert dataset_num_features == dataset_num_features_ood

    num_sample = len(dataset)
    num_train = int(num_sample * train_per)
    indices = torch.randperm(num_sample)
    idx_train = torch.sort(indices[:num_train])[0]
    idx_test = torch.sort(indices[num_train:])[0]

    dataset_train = dataset[idx_train]
    dataset_test = dataset[idx_test]
    dataset_ood = dataset_ood[: len(dataset_test)]

    data_list_train = []
    idx = 0
    for data in dataset_train:
        data.y = 0
        data['idx'] = idx
        idx += 1
        data_list_train.append(data)

    if need_str_enc:
        data_list_train = init_structural_encoding(data_list_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    fb_keys = [key for key in dataset[0].keys if key.find('treePHLayer')>=0]
    dataloader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True, follow_batch=fb_keys)

    data_list_test = []
    for data in dataset_test:
        data.y = 0
        data.edge_attr = None
        data_list_test.append(data)

    for data in dataset_ood:
        data.y = 1
        data.edge_attr = None
        data_list_test.append(data)

    if need_str_enc:
        data_list_test = init_structural_encoding(data_list_test, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    if pre:
        max_degree=data_list_test[0].deg_x.shape[1]-2
        for data in data_list_test:
            if data.x is not None:
                deg = degree(data.edge_index[0], data.x.shape[0], dtype=torch.long)
            else:
                deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
            deg = deg.view((-1, 1))
            max_deg = torch.tensor(max_degree, dtype=deg.dtype)
            deg = torch.min(deg, max_deg).view(-1)
            import torch.nn.functional as F
            onehot_deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)
            if data.x is not None:
                data.deg_x = torch.cat([data.x, onehot_deg.to(data.x.dtype)], dim=-1)
            else:
                data.deg_x = onehot_deg

    fb_keys = [key for key in dataset[0].keys if key.find('treePHLayer') >= 0]
    dataloader_test = DataLoader(data_list_test, batch_size=args.batch_size_test, shuffle=True, follow_batch=fb_keys)

    if pre:
        meta = {'num_feat':dataset_num_features, 'num_train':len(dataset_train), 'deg_x':dataset_train[0].deg_x.shape[1],
                'num_test':len(dataset_test), 'num_ood':len(dataset_ood)}
    else:
        meta = {'num_feat': dataset_num_features, 'num_train': len(dataset_train),
                'num_test': len(dataset_test), 'num_ood': len(dataset_ood)}
    return dataloader, dataloader_test, meta


