import os

# ============================================================
# MUST be set before torch / numpy / scipy / mmcv
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
import torch.utils
import torch.utils.data
from torch_geometric.data import InMemoryDataset, Data
import mmcv
import rdkit
from rdkit import Chem
from tqdm import tqdm
import os.path as osp
from scipy import sparse as sp
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx
from utils.smiles2graph import smile2graph4drugood

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


class SmileToGraph(object):
    # Adapt from https://github.com/tencent-ailab/DrugOOD/blob/main/drugood/datasets/pipelines/formating.py
    """Transform smile input to graph format"""

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = self.smile2graph(results[key])
        return results

    def get_atom_features(self, atom):
        # The usage of features is along with the Attentive FP.
        feature = np.zeros(4)
        # Symbol
        symbol = atom.GetSymbol()
        # 6: C, 7: N, 8: O, 9: F etc.
        symbol_list = ['C', 'N', 'O', 'F']
        if symbol in symbol_list:
            loc = symbol_list.index(symbol)
            feature[loc] = 1

        return feature

    def get_bond_features(self, bond):
        feature = np.zeros(10)

        # bond type
        type = bond.GetBondType()
        bond_type_list = [rdkit.Chem.rdchem.BondType.SINGLE,
                          rdkit.Chem.rdchem.BondType.DOUBLE,
                          rdkit.Chem.rdchem.BondType.TRIPLE,
                          rdkit.Chem.rdchem.BondType.AROMATIC]
        if type in bond_type_list:
            loc = bond_type_list.index(type)
            feature[0 + loc] = 1
        else:
            print("Wrong type of bond. Please check before feturization.")
            raise RuntimeError

        # conjugation
        conj = bond.GetIsConjugated()
        feature[4] = conj

        # ring
        ring = bond.IsInRing()
        feature[5] = ring

        # stereo
        stereo = bond.GetStereo()
        stereo_list = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                       rdkit.Chem.rdchem.BondStereo.STEREOANY,
                       rdkit.Chem.rdchem.BondStereo.STEREOZ,
                       rdkit.Chem.rdchem.BondStereo.STEREOE]
        if stereo in stereo_list:
            loc = stereo_list.index(stereo)
            feature[6 + loc] = 1
        else:
            print("Wrong stereo type of bond. Please check before featurization.")
            raise RuntimeError

        return feature

    def smile2graph(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if (mol is None):
            return None
        src = []
        dst = []
        atom_feature = []
        bond_feature = []
        try:
            for atom in mol.GetAtoms():
                one_atom_feature = self.get_atom_features(atom)

                atom_feature.append(one_atom_feature)

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                one_bond_feature = self.get_bond_features(bond)
                src.append(i)
                dst.append(j)
                bond_feature.append(one_bond_feature)
                src.append(j)
                dst.append(i)
                bond_feature.append(one_bond_feature)

            src = torch.tensor(src).long()
            dst = torch.tensor(dst).long()
            edge_index = torch.vstack([src, dst])
            atom_feature = np.array(atom_feature)
            bond_feature = np.array(bond_feature)
            atom_feature = torch.tensor(atom_feature).float()
            bond_feature = torch.tensor(bond_feature).float()
            graph_cur_smile = Data(x=atom_feature, edge_index=edge_index, edge_attr=bond_feature)
            return graph_cur_smile

        except RuntimeError:
            return None
class DrugOOD(InMemoryDataset):
    splits = ['iid', 'ood', 'mixed']

    def __init__(self, root, mode='iid', d_type='size', d_name='ec50', transform=None, pre_transform=None,
                 pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        self.d_name = d_name
        self.d_type = d_type
        self.smile2graph = SmileToGraph(['smiles'])
        super(DrugOOD, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('drugood_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_dir(self):
        return osp.join(self.root, f'{self.d_name}/{self.d_type}')

    @property
    def raw_file_names(self):
        return ['lbap_general_{}_{}.json'.format(self.d_name, self.d_type)]

    @property
    def processed_dir(self):

        return osp.join(self.root, f'{self.d_name}/{self.d_type}')

    @property
    def processed_file_names(self):
        return [f'drugood_{self.mode}.pt']

    def download(self):
        if not osp.exists(osp.join(self.raw_dir, self.raw_file_names[0])):
            print("raw data of `DrugOOD` doesn't exist, please redownload from our github.")
            raise FileNotFoundError

    def process(self):
        if self.mode == 'iid':
            raw_dataset = mmcv.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['train']
            num = 6000
        elif self.mode == 'ood':
            raw_dataset = mmcv.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['ood_test']
            num = 5000
        elif self.mode == 'mixed':
            raw_dataset = mmcv.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['ood_val']
            num = 1000
        data_list = []
        for idx, case in enumerate(tqdm(raw_dataset[:num], desc='Iteration')):
            # case = self.smile2graph(case)
            # case['smiles'].y = torch.tensor(case['cls_label'], dtype=torch.long).unsqueeze(dim=0)
            # case['smiles'].idx = idx
            # data = case['smiles']
            smiles = case['smiles']
            x, edge_index, edge_attr = smile2graph4drugood(smiles)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            data.y = torch.tensor(case['cls_label'], dtype=torch.long).unsqueeze(dim=0)
            data.idx = idx
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        idx = self.processed_file_names.index('drugood_{}.pt'.format(self.mode))

        torch.save(self.collate(data_list), self.processed_paths[idx])
class DrugOOD_ET(InMemoryDataset):
    splits = ['iid', 'ood', 'mixed']

    def __init__(self, root, mode='iid', d_type='size', d_name='ec50', need_size = 5000, transform=None, pre_transform=None,
                 pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        self.d_name = d_name
        self.d_type = d_type
        self.size = need_size
        # self.smile2graph = SmileToGraph(['smiles'])
        super(DrugOOD_ET, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('drugood_ET_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_dir(self):
        return osp.join(self.root, f'{self.d_name}/{self.d_type}')

    @property
    def raw_file_names(self):
        return ['lbap_general_{}_{}.json'.format(self.d_name, self.d_type), 'drugood_{}_ecloud_patch.pt'.format(self.mode)]

    @property
    def processed_dir(self):

        return osp.join(self.root, f'{self.d_name}/{self.d_type}')

    @property
    def processed_file_names(self):
        return [f'drugood_ET_{self.mode}.pt']

    def download(self):
        if not osp.exists(osp.join(self.raw_dir, self.raw_file_names[0])):
            print("raw data of `DrugOOD` doesn't exist, please redownload from our github.")
            raise FileNotFoundError

    def process(self):
        if self.mode == 'iid':
            raw_dataset = mmcv.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['train']
            ecloud_datas = torch.load(osp.join(self.raw_dir, self.raw_file_names[1]))
            num = self.size
        elif self.mode == 'ood':
            raw_dataset = mmcv.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['ood_test']
            ecloud_datas = torch.load(osp.join(self.raw_dir, self.raw_file_names[1]))
            num = self.size
        elif self.mode == 'mixed':
            raw_dataset = mmcv.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['ood_val']
            ecloud_datas = torch.load(osp.join(self.raw_dir, self.raw_file_names[1]))
            num = self.size
        data_list = []
        now_num = 0
        for idx, case in enumerate(tqdm(raw_dataset, desc='Iteration')):
            if(idx >= len(ecloud_datas)):
                break
            ecloud_data = ecloud_datas[case['smiles']]
            if(ecloud_data is None):
                continue
            # case = self.smile2graph(case)
            smiles = case['smiles']
            x, edge_index, edge_attr = smile2graph4drugood(smiles)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            data.y = torch.tensor(case['cls_label'], dtype=torch.long).unsqueeze(dim=0)
            data.idx = idx
            data.ecloud_data = ecloud_data
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            now_num += 1
            if(now_num == num):
                break

        idx = self.processed_file_names.index('drugood_ET_{}.pt'.format(self.mode))
        torch.save(self.collate(data_list), self.processed_paths[idx])

def Get_OOD_DataLoader(args, train_per=0.9, need_str_enc=True, pre=True):
    d_name, d_type =  args.dataset_name.split('_')
    if pre:
        from Dataload.codingTree import GraphTransform
        degreeDim = 10
        # degreeDim = 10 if args.DS in ['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K'] else 100
        pre_transform = GraphTransform(args.t_depth, degreeDim).transform

        import time
        start_time = time.time()
        dataset = DrugOOD_ET(osp.join(args.data_root, args.dataset), mode='iid', d_name=d_name, d_type=d_type, need_size=args.iid_size, pre_transform=pre_transform)
        dataset_ood = DrugOOD_ET(osp.join(args.data_root, args.dataset), mode='ood', d_name=d_name, d_type=d_type, need_size=args.ood_size, pre_transform=pre_transform)

        end_time = time.time()
        runtime = end_time - start_time
        print(f"Coding t construction amd computer ecloud time: {runtime:.2f} s")

    else:
        dataset = DrugOOD(osp.join(args.data_root, args.dataset), mode='iid', d_name=d_name, d_type=d_type)
        dataset_ood = DrugOOD(osp.join(args.data_root, args.dataset), mode='ood', d_name=d_name, d_type=d_type)

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

    fb_keys = [key for key in dataset[0].keys if key.find('tPHLayer') >= 0]
    dataloader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True, follow_batch=fb_keys)

    data_list_test = []
    for data in dataset_test:
        data.y = 0
        #data.edge_attr = None
        data_list_test.append(data)

    for data in dataset_ood:
        data.y = 1
        #data.edge_attr = None
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

    fb_keys = [key for key in dataset[0].keys if key.find('tPHLayer') >= 0]
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
    parser.add_argument('--dataset', default="DrugOOD", type=str,
                        help='dataset name (plym-, ogbg-)')
    parser.add_argument('--data-root', default="../data/", type=str)
    parser.add_argument('---dataset_name', default='ec50_assay', type=str)
    # training
    parser.add_argument("--t_depth", type=int, default=5)

    # augmentation
    parser.add_argument("-rw_dim", type=int, default=16)
    parser.add_argument("-dg_dim", type=int, default=16)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-batch_size_test", type=int, default=128)
    parser.add_argument("-iid_size", type=int, default=8000)
    parser.add_argument("-ood_size", type=int, default=5000)
    args = parser.parse_args()



    dataset_name = ['ec50_assay', 'ec50_scaffold', 'ec50_size',
                    'ic50_assay', 'ic50_scaffold', 'ic50_size']

    for i in range(len(dataset_name)):
        print(dataset_name[i])
        args.dataset_name = dataset_name[i]
        dataloader, dataloader_test, meta = Get_OOD_DataLoader(args)
        print('finish')


