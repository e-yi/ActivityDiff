import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem import Lipinski

import sascorer


def get_Inchi(mol_list):
    return np.array([Chem.MolToInchi(mol) for mol in mol_list])


def get_Inchi_key(mol_list):
    return np.array([Chem.MolToInchiKey(mol) for mol in mol_list])


def get_num_H_acceptors(mol_list):
    return np.array([Lipinski.NumHAcceptors(mol) for mol in mol_list])


def get_num_H_donors(mol_list):
    return np.array([Lipinski.NumHDonors(mol) for mol in mol_list])


def get_num_rotatable_bonds(mol_list):
    return np.array([Lipinski.NumRotatableBonds(mol) for mol in mol_list])


def get_molecular_weights(mol_list):
    return np.array([Descriptors.ExactMolWt(mol) for mol in mol_list])


def get_logp(mol_list):
    return np.array([Descriptors.MolLogP(mol) for mol in mol_list])


def get_tpsa(mol_list):
    return np.array([Descriptors.TPSA(mol) for mol in mol_list])


def get_qed(mol_list):
    return np.array([QED.qed(mol) for mol in mol_list])


def get_sa(mol_list):
    return np.array([sascorer.calculateScore(mol) for mol in mol_list])


def get_lipinski_five(mol_list):
    mol_wt = get_molecular_weights(mol_list)
    hbd = get_num_H_donors(mol_list)
    hba = get_num_H_acceptors(mol_list)
    logp = get_logp(mol_list)
    num_rotatable_bonds = get_num_rotatable_bonds(mol_list)

    score = ((mol_wt < 500).astype(int) +
             (hbd < 5).astype(int) +
             (hba < 10).astype(int) +
             np.bitwise_and(-2 < logp, logp < 5).astype(int) +
             (num_rotatable_bonds < 10).astype(int))

    return score


def test_molecular_descriptors():
    mol_smiles = ['CCO', 'CC(=O)OC1=CC=CC=C1C(=O)O', 'CC(C)(C)C1=CC=CC=C1C(=O)O']
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in mol_smiles]

    print(mol_smiles)

    # Test each function and print their outputs
    print("Testing get_Inchi:")
    print(get_Inchi(mol_list))

    print("Testing get_Inchi_key:")
    print(get_Inchi_key(mol_list))

    print("Testing get_num_H_acceptors:")
    print(get_num_H_acceptors(mol_list))

    print("Testing get_num_H_donors:")
    print(get_num_H_donors(mol_list))

    print("Testing get_num_rotatable_bonds:")
    print(get_num_rotatable_bonds(mol_list))

    print("Testing get_molecular_weights:")
    print(get_molecular_weights(mol_list))

    print("Testing get_logp:")
    print(get_logp(mol_list))

    print("Testing get_tpsa:")
    print(get_tpsa(mol_list))

    print("Testing get_qed:")
    print(get_qed(mol_list))

    print("Testing get_sa:")
    print(get_sa(mol_list))

    print("Testing test_lipinski_five")
    print(get_lipinski_five(mol_list))


if __name__ == '__main__':
    test_molecular_descriptors()
