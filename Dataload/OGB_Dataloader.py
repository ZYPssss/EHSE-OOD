import os

# ============================================================
# MUST be set before torch / numpy / scipy / mmcv
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import shutil
from ogb.graphproppred import PygGraphPropPredDataset
import pandas as pd
import argparse
import torch.utils
import torch.utils.data
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
from scipy import sparse as sp
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx
import ogb
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg

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


class PygGraphPropPredDataset_ET(InMemoryDataset):
    def __init__(self, name, root = 'dataset', transform=None, pre_transform = None, meta_dict = None, processed_filename='data.pt'):
        self.name = name ## original name, e.g., ogbg-molhiv
        self.processed_filename = processed_filename

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            # master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
            master_path = osp.join(
                osp.dirname(ogb.__file__),
                'graphproppred',
                'master.csv'
            )
            master = pd.read_csv(master_path, index_col=0)
            ## change for tree1017
            # master = pd.read_csv('/ifs/home/zhangyuanpeng/anaconda3/envs/py37/lib/python3.7/site-packages/ogb/graphproppred/master.csv',
            #     index_col=0)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'

        super(PygGraphPropPredDataset_ET, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

    def get_Smiles(self):
        original_data = pd.read_csv(
            os.path.join(self.original_root, self.dir_name, 'mapping', 'mol.csv.gz'),
            compression='gzip'
        )
        return original_data.smiles

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return ['OGB_ET.pt']
        # return self.processed_filename

    def download(self):
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        ### read pyg graph list
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        data_list = read_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)
        Smlies = self.get_Smiles()

        eclouds = torch.load(os.path.join(self.original_root, self.dir_name, "ogb_all_ecloud_patch.pt"))
        data_list_new = []
        for i, g in enumerate(data_list):
            ecloud_data = eclouds[Smlies[i]]
            if(ecloud_data is None):
                continue
            else:
                g.ecloud_data = ecloud_data
                data_list_new.append(g)
        data_list = data_list_new
        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values

            has_nan = np.isnan(graph_label).any()


            for i, g in enumerate(data_list):
                if 'classification' in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

class PygGraphPropPredDataset_ET_split(InMemoryDataset):
    def __init__(self, name, root = 'dataset', mode = 'iid', transform=None, pre_transform = None, meta_dict = None, processed_filename='data.pt'):
        self.name = name ## original name, e.g., ogbg-molhiv
        self.processed_filename = processed_filename
        self.mode = mode

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            # master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
            master_path = osp.join(
                osp.dirname(ogb.__file__),
                'graphproppred',
                'master.csv'
            )
            master = pd.read_csv(master_path, index_col=0)
            ## change for tree1017
            # master = pd.read_csv('/ifs/home/zhangyuanpeng/anaconda3/envs/py37/lib/python3.7/site-packages/ogb/graphproppred/master.csv',
            #     index_col=0)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'

        super(PygGraphPropPredDataset_ET_split, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

    def get_Smiles(self):
        original_data = pd.read_csv(
            os.path.join(self.original_root, self.dir_name, 'mapping', 'mol.csv.gz'),
            compression='gzip'
        )
        return original_data.smiles

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return [f'OGB_ET_{self.mode}.pt']
        # return self.processed_filename

    def download(self):
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        ### read pyg graph list
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        data_list = read_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)
        split_idx = self.get_idx_split()
        if(self.mode == 'iid'):

            train_indices = split_idx['train'].tolist()  # 或 .numpy() 也行，但 .tolist() 是 list
            data_list = [data_list[i] for i in train_indices]
        elif(self.mode == 'ood'):
            train_indices = split_idx['test'].tolist()  # 或 .numpy() 也行，但 .tolist() 是 list
            data_list = [data_list[i] for i in train_indices]

        Smlies = self.get_Smiles()

        eclouds = torch.load(os.path.join(self.original_root, self.dir_name, "ogb_all_ecloud_patch.pt"))
        data_list_new = []
        for i, g in enumerate(data_list):
            ecloud_data = eclouds[Smlies[i]]
            if(ecloud_data is None):
                continue
            else:
                g.ecloud_data = ecloud_data
                data_list_new.append(g)
        data_list = data_list_new
        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values

            has_nan = np.isnan(graph_label).any()


            for i, g in enumerate(data_list):
                if 'classification' in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])





def Get_OOD_DataLoader(args, train_per=0.9, need_str_enc=True, pre=True):

    if pre:
        from Dataload.codingTree import GraphTransform
        degreeDim = 10
        # degreeDim = 10 if args.DS in ['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K'] else 100
        pre_transform = GraphTransform(args.tree_depth, degreeDim).transform

        import time
        start_time = time.time()

        if (args.ood_dataset_name is None):
            dataset = PygGraphPropPredDataset_ET_split(name=args.dataset_name, root=osp.join(args.data_root, args.dataset), mode = 'iid',
                                                 pre_transform=pre_transform)
            dataset.data.x = dataset.data.x.type(torch.float32)

            dataset_ood = PygGraphPropPredDataset_ET_split(name=args.dataset_name, root=osp.join(args.data_root, args.dataset), mode='ood',
                                                                    pre_transform=pre_transform)
            dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)


        else:
            dataset = PygGraphPropPredDataset_ET(name=args.dataset_name, root=osp.join(args.data_root, args.dataset), pre_transform=pre_transform)
            dataset.data.x = dataset.data.x.type(torch.float32)
            dataset_ood = PygGraphPropPredDataset_ET(name=args.dataset_name, root=osp.join(args.data_root, args.dataset), pre_transform=pre_transform)
            dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)

        end_time = time.time()
        runtime = end_time - start_time
        print(f"Coding tree construction amd computer ecloud time: {runtime:.2f} s")

    else:

        if (args.ood_dataset_name is None):
            dataset = PygGraphPropPredDataset(name=args.dataset_name, root=osp.join(args.data_root, args.dataset))
            dataset.data.x = dataset.data.x.type(torch.float32)
            split_idx = dataset.get_idx_split()
            dataset_ood = [dataset[x.item()] for x in split_idx['test']]
            dataset = [dataset[x.item()] for x in split_idx['train']]

        else:
            dataset = PygGraphPropPredDataset(name=args.dataset_name, root=osp.join(args.data_root, args.dataset))
            dataset.data.x = dataset.data.x.type(torch.float32)
            dataset_ood = PygGraphPropPredDataset(name=args.dataset_name, root=osp.join(args.data_root, args.dataset))
            dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)

    dataset_num_features = dataset.num_node_features
    dataset_num_features_ood = dataset_ood.num_node_features
    edge_feat = dataset.num_edge_features
    assert dataset_num_features == dataset_num_features_ood

    num_sample = len(dataset)
    num_train = int(num_sample * train_per)
    indices = torch.randperm(num_sample)
    idx_train = torch.sort(indices[:num_train])[0]
    idx_test = torch.sort(indices[num_train:])[0]
    dataset_train = dataset[idx_train]
    dataset_test = dataset[idx_test]
    dataset_ood = dataset_ood[: len(dataset_test)]

    # perm_idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
    # dataset = dataset[perm_idx]
    #
    # perm_idx = torch.randperm(len(dataset_ood), generator=torch.Generator().manual_seed(0))
    # dataset_ood = dataset_ood[perm_idx]
    #
    # n_train_data, n_val_data, n_in_test_data, n_out_test_data = 1000, 500, 500, 500
    # dataset_train, dataset_test = dataset[:n_train_data], dataset[n_train_data:n_train_data + n_in_test_data]
    # val_dataset = dataset_ood[-n_val_data:]
    # dataset_ood = dataset_ood[:n_out_test_data]

    data_list_train = []
    idx = 0
    for data in dataset_train:
        data.y = 0
        data['idx'] = idx
        idx += 1
        data_list_train.append(data)

    if need_str_enc:
        data_list_train = init_structural_encoding(data_list_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    fb_keys = [key for key in dataset[0].keys if key.find('treePHLayer') >= 0]
    dataloader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True, follow_batch=fb_keys)

    data_list_test = []
    for data in dataset_test:
        data.y = 0
        data_list_test.append(data)

    for data in dataset_ood:
        data.y = 1
        data_list_test.append(data)

    if need_str_enc:
        data_list_test = init_structural_encoding(data_list_test, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    if pre:
        # max_degree = data_list_test[0].deg_x.shape[1] - 2
        degree_dim = (
                data_list_test[0].deg_x.shape[1]
                - data_list_test[0].x.shape[1]
        )
        max_degree = degree_dim - 1
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
        meta = {'num_feat': dataset_num_features, 'edge_feat': edge_feat, 'num_train': len(dataset_train),
                'deg_x': dataset_train[0].deg_x.shape[1],
                'num_test': len(dataset_test), 'num_ood': len(dataset_ood)}
    else:
        meta = {'num_feat': dataset_num_features, 'edge_feat': edge_feat, 'num_train': len(dataset_train),
                'num_test': len(dataset_test), 'num_ood': len(dataset_ood)}
    return dataloader, dataloader_test, meta




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', default="OGB", type=str,
                        choices=['DrugOOD', 'GOOD'],
                        help='dataset name (plym-, ogbg-)')
    parser.add_argument('--data-root', default="../data/", type=str)
    parser.add_argument('---dataset_name', default='ogbg-molbbbp', type=str)
    parser.add_argument('---ood_dataset_name', default=None, type=str)

    # training
    parser.add_argument("--tree_depth", type=int, default=5)

    # augmentation
    parser.add_argument("-rw_dim", type=int, default=16)
    parser.add_argument("-dg_dim", type=int, default=16)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-batch_size_test", type=int, default=128)
    args = parser.parse_args()



    dataset_name = ['ogbg-molbbbp', 'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molsider']

    for i in range(len(dataset_name)):
        args.dataset_name = dataset_name[i]
        dataloader, dataloader_test, meta = Get_OOD_DataLoader(args)
        print('finish')
        break


