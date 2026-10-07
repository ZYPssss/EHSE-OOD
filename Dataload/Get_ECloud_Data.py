import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, BASE_DIR)
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from rdkit import Chem
from tqdm import tqdm
import argparse
import pandas as pd
from utils.chem import gen_geom_with_rdkit, set_mol_position
from ecloud_utils.xtb_density import CDCalculator, interplot_ecloud
from ecloud_utils.rotation import uniformRandomRotation, rotate
from ecloud_utils.grid import BuildGridCenters
import numpy as np
from model.ecloud_encoder import ElectronEncoder
import mmcv
import shutil

print(shutil.which("/data4t/zhangyuanpeng/anaconda3/envs/py37/bin/xtb"))
calculater = CDCalculator(xtb_command="/data4t/zhangyuanpeng/anaconda3/envs/py37/bin/xtb")

E_coder = ElectronEncoder(32)


def protocol(mode=32):
    '''
    Define the grid grid_protocol, including grid size, resolution, and grid centers
        grid size: 32 or 64
        resolution: 0.5 or 0.2
        grid centers: the center of the grid
    Input:
        mode: grid mode, 32 or 64
    Output:
        {'grids':grids, 'N':N}
    '''
    size = mode
    N = [size, size, size]
    if mode == 32:
        resolution = 0.5
        llc = (np.zeros(3) - float(size * resolution / 2)) + resolution / 2
        grids = BuildGridCenters(llc, N, resolution)
    elif mode == 64:
        resolution = 0.2
        llc = (np.zeros(3) - float(size * resolution / 2)) + resolution / 2
        grids = BuildGridCenters(llc, N, resolution)

    return {'grids': grids, 'N': N}

def get_ecloud(smi, rand_rotate=True, grid_protocol=protocol()):
    lig_mol = Chem.MolFromSmiles(smi)
    lig_mol = gen_geom_with_rdkit(lig_mol)
    lig_coords = lig_mol.GetConformer().GetPositions()
    lig_center = lig_coords.mean(axis=0)
    if rand_rotate:
        rrot = uniformRandomRotation()  # Rotation
        lig_coords = rotate(lig_coords, rrot, center=lig_center)
    rotated_lig_mol = set_mol_position(lig_mol, lig_coords)
    lig_ecloud = calculater.calculate(rotated_lig_mol)

    lig_grids = grid_protocol['grids'] + lig_center
    lig_density = interplot_ecloud(lig_ecloud, lig_grids.transpose(3, 0, 1, 2)).reshape(grid_protocol['N'])
    return lig_density, lig_coords, lig_center


def get_drug_ecloud(smi):
    #smi = smiles[index]
    try:
        ecloud, lig_coords, lig_center = get_ecloud(smi)
    except Exception as e:
        print(e)
        ecloud = None
        lig_coords = None
        lig_center = None
    return smi, ecloud, lig_coords, lig_center

def get_drug_ecloud_patches_data(smi, resolution=0.5):

    smi, density, atom_positions, lig_center = get_drug_ecloud(smi)
    if density is None:
        return None
    else:
        patches = []
        for pos in atom_positions:
            patch = sample_patch(
                density,
                pos,
                lig_center,
                resolution
            )
            patches.append(patch)

        patches = torch.tensor(
            np.stack(patches),
            dtype=torch.float32
        ).unsqueeze(1)
        # electron_feature = encoder(patches)
        return patches
# -----------------------------
# 提取每个原子的局部电子云Patch
# -----------------------------
def sample_patch(density, atom_pos, lig_center, resolution=0.5, patch_size=5):

    grid_size = density.shape[0]
    half = grid_size * resolution / 2
    # 转到局部坐标
    coord = atom_pos - lig_center
    idx = ((coord + half) / resolution).astype(int)
    idx = np.clip(idx, 0, grid_size - 1)
    r = patch_size // 2
    x, y, z = idx
    patch = np.zeros((patch_size, patch_size, patch_size),
                     dtype=np.float32)
    xs = max(0, x-r)
    xe = min(grid_size, x+r+1)
    ys = max(0, y-r)
    ye = min(grid_size, y+r+1)
    zs = max(0, z-r)
    ze = min(grid_size, z+r+1)

    patch[
        0:xe-xs,
        0:ye-ys,
        0:ze-zs
    ] = density[xs:xe, ys:ye, zs:ze]

    return patch

def process_smiles_parallel(smiles_list):
    ecloud_patchs = {}
    for i in tqdm(range(len(smiles_list))):
    # Collect successful results
        patches = get_drug_ecloud_patches_data(smiles_list[i])
        ecloud_patchs[smiles_list[i]]=patches

def get_save_DrugOOD_Ecloud_data_by_dataset_name(dataset_name, mode, num):

    d_name, d_type = dataset_name.split('_')
    if(mode == 'iid'):
        raw_dataset = mmcv.load('../data/DrugOOD/{}/{}/lbap_general_{}_{}.json'.format(d_name, d_type, d_name, d_type))['split']['train']
    elif(mode == 'ood'):
        raw_dataset = mmcv.load('../data/DrugOOD/{}/{}/lbap_general_{}_{}.json'.format(d_name, d_type, d_name, d_type))['split']['ood_test']

    ecloud_patchs = {}
    if (num == -1):
        num = len(raw_dataset)
    for idx, case in enumerate(tqdm(raw_dataset[:num], desc='Iteration')):
        smi = case['smiles']

        patches = get_drug_ecloud_patches_data(smi)
        ecloud_patchs[smi] = patches

    torch.save(ecloud_patchs, "../data/DrugOOD/{}/{}/drugood_{}_ecloud_patch.pt".format(d_name, d_type, mode))
    x = torch.load("../data/DrugOOD/{}/{}/drugood_{}_ecloud_patch.pt".format(d_name, d_type, mode))
    print("Successfully saved {} {} processed molecules".format(dataset_name, mode))

def get_save_OGB_Ecloud_data_by_dataset_name(dataset_name, mode, num):
    dataset = PygGraphPropPredDataset(dataset_name, root='../data/OGB')
    split_idx = dataset.get_idx_split()
    dataset_dir = dataset_name.replace('-', '_')
    original_data = pd.read_csv(
        os.path.join('../data/OGB', dataset_dir,  'mapping', 'mol.csv.gz'),
        compression='gzip'
    )

    if(mode == 'iid'):
        raw_dataset = [original_data.smiles[x.item()] for x in split_idx['train']]
    elif(mode == 'ood'):
        raw_dataset = [original_data.smiles[x.item()] for x in split_idx['test']]
    elif(mode == 'all'):
        raw_dataset = original_data.smiles

    ecloud_patchs = {}
    if(num==-1):
        num = len(raw_dataset)

    for idx, case in enumerate(tqdm(raw_dataset[:num], desc='Iteration')):
        smi = case

        patches = get_drug_ecloud_patches_data(smi)
        ecloud_patchs[smi] = patches

    torch.save(ecloud_patchs, "../data/OGB/{}/ogb_{}_ecloud_patch.pt".format(dataset_dir, mode))
    x = torch.load("../data/OGB/{}/ogb_{}_ecloud_patch.pt".format(dataset_dir, mode))
    print("Successfully saved {} {} processed molecules".format(dataset_dir, mode))


def get_save_all_DrugOOD_Ecloud_data():
    dataset_name = ['ec50_assay', 'ec50_scaffold', 'ec50_size',
                    'ic50_assay', 'ic50_scaffold', 'ic50_size']

    for i in range(len(dataset_name)):
        print('----------------load dataset: {}-----------------'.format(dataset_name[i]))
        get_save_DrugOOD_Ecloud_data_by_dataset_name(dataset_name[i], 'iid', -1)
        get_save_DrugOOD_Ecloud_data_by_dataset_name(dataset_name[i], 'ood', -1)

def get_save_all_OGB_Ecloud_data():
    dataset_name = ['ogbg-molbbbp', 'ogbg-molbace', 'ogbg-molsider', 'ogbg-molhiv']
    dataset_name = ['ogbg-molsider']

    for i in range(len(dataset_name)):
        print('----------------load dataset: {}-----------------'.format(dataset_name[i]))
        # get_save_OGB_Ecloud_data_by_dataset_name(dataset_name[i], 'iid', -1)
        # get_save_OGB_Ecloud_data_by_dataset_name(dataset_name[i], 'ood', -1)
        get_save_OGB_Ecloud_data_by_dataset_name(dataset_name[i], 'all', -1)


if __name__ == '__main__':
    get_save_all_OGB_Ecloud_data()



