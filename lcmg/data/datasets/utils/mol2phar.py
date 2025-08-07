import os

import numpy as np
import torch
from rdkit import RDConfig, RDLogger
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import ChemicalFeatures
from scipy.spatial.distance import cdist

RDLogger.DisableLog('rdApp.*')

# treat Hydrophobe and LumpedHydrophobe the same
RDKIT_PHAR_MAPPING = {'Aromatic': 0, 'Hydrophobe': 1, 'LumpedHydrophobe': 1, 'PosIonizable': 2, 'NegIonizable': 3,
                      'Acceptor': 4, 'Donor': 5, 'Exclusion': 6, 'Unknown': 7}
DEFAULT_SIZE_MAPPING = None  # <del>按实际直径分bin</del> 不分了，直接加噪声

__table = Chem.GetPeriodicTable()


def get_atom_radius(atom_symbol):
    """
    Get the covalent radius of an atom.
    The values are from the blue obelisk project (are they outdated?). https://github.com/rdkit/rdkit/issues/1831
    :param atom_symbol:
    :return:
    """
    return __table.GetRcovalent(__table.GetAtomicNumber(atom_symbol))


def get_rdkit_phar(mol, fdef_name=os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')):
    """
    return phars: [(phar_type:int, atom_indicis:List[int])]
    """

    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    phars = []
    feats = factory.GetFeaturesForMol(mol)
    for f in feats:
        phar = f.GetFamily()
        atom_idx_list = f.GetAtomIds()
        atom_idx_list = list(sorted(atom_idx_list))
        if phar not in RDKIT_PHAR_MAPPING:
            # log.debug(f'{phar} will be labeled as `Unknown`')
            phar = 'Unknown'
            continue
        phar_type_id = RDKIT_PHAR_MAPPING[phar]
        phars.append((phar_type_id, atom_idx_list))  # some pharmacophore feature

    return phars


def mol2phar(mol, fdef_name=os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'), ):
    """
    input mol
    output phar features, which include phar_type, phar_pos, phar_size(radius),
    todo exclusion
    :param mol:
    :param fdef_name:
    :return:
    """
    mol = Chem.Mol(mol)

    phars = get_rdkit_phar(mol, fdef_name)

    positions = mol.GetConformers()[0].GetPositions()

    dist = cdist(positions, positions)

    # account for the atom radius
    atom_radius = [get_atom_radius(mol.GetAtomWithIdx(i).GetSymbol()) for i in range(mol.GetNumAtoms())]
    atom_radius = np.array(atom_radius)
    dist += atom_radius.reshape(1, -1) + atom_radius.reshape(-1, 1)

    phar_type = []
    phar_pos = []
    phar_size = []
    for phar_type_id, atom_idx_list in phars:
        phar_type.append(phar_type_id)
        phar_pos.append(positions[atom_idx_list].mean(0))
        phar_size.append(dist[atom_idx_list, :][:, atom_idx_list].max() / 2)

    return torch.LongTensor(phar_type), torch.FloatTensor(np.array(phar_pos)), torch.FloatTensor(phar_size)


def test():
    smiles = "C1=CC=C(C=C1)C(=O)CCCC"

    mol = Chem.MolFromSmiles(smiles)

    # 进行三维构象优化
    Chem.EmbedMolecule(mol)

    res = mol2phar(mol)

    print(res)


if __name__ == '__main__':
    test()
