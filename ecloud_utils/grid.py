import numpy as np
from rdkit import Chem
try:
    from .htmd_utils import _getChannelRadii
except:
    from ecloud_utils.htmd_utils import _getChannelRadii

def BuildGridCenters(llc, N, step):
    """
    llc: lower left corner
    N: number of cells in each direction
    step: step size
    """

    if type(step) == float:
        xrange = [llc[0] + step * x for x in range(0, N[0])]
        yrange = [llc[1] + step * x for x in range(0, N[1])]
        zrange = [llc[2] + step * x for x in range(0, N[2])]
    elif type(step) == list or type(step) == tuple:
        xrange = [llc[0] + step[0] * x for x in range(0, N[0])]
        yrange = [llc[1] + step[1] * x for x in range(0, N[1])]
        zrange = [llc[2] + step[2] * x for x in range(0, N[2])]

    centers = np.zeros((N[0], N[1], N[2], 3))
    for i, x in enumerate(xrange):
        for j, y in enumerate(yrange):
            for k, z in enumerate(zrange):
                centers[i, j, k, :] = np.array([x, y, z])
    return centers

resolution = 1.
size = 24
N = [size, size, size]
llc = (np.zeros(3) - float(size * 1. / 2))
# Now, the box is 24×24×24 A^3
expanded_pcenters = BuildGridCenters(llc, N, resolution)




def get_aromatic_groups(in_mol):
    """
    Obtain groups of aromatic rings
    """
    groups = []
    ring_atoms = in_mol.GetRingInfo().AtomRings()
    for ring_group in ring_atoms:
        if all([in_mol.GetAtomWithIdx(x).GetIsAromatic() for x in ring_group]):
            groups.append(ring_group)
    return groups

def generate_occpy(mol):
    """
    Only Calculates the occupancy
    """
    coords = mol._coords[: , : , 0]
    n_atoms = len(coords)
    lig_center = mol.getCenter()
    

def generate_sigmas(mol):
    """
    Calculates sigmas for elements as well as pharmacophores.
    Returns sigmas, coordinates and center of ligand.
    """
    coords = mol._coords[: , : , 0]
    n_atoms = len(coords)
    lig_center = mol.getCenter()

    # Calculate all the channels
    multisigmas = _getChannelRadii(mol) 

    return multisigmas, coords, lig_center


