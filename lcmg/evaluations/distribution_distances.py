import numpy as np
from scipy.spatial import distance


def get_atom_type_jsd(mol_list, *, freq_atom_type, at2idx, **kwargs):
    """
    计算原子类型的JS距离
    """
    if len(mol_list) == 0:
        return np.nan

    atom_type_distribution = np.zeros_like(freq_atom_type, dtype=float)

    for mol in mol_list:
        for atom in mol.GetAtoms():
            atom_type_distribution[at2idx[atom.GetSymbol()]] += 1

    atom_type_distribution /= atom_type_distribution.sum()

    js_dist = distance.jensenshannon(atom_type_distribution, freq_atom_type)

    return js_dist


def get_bond_type_jsd(mol_list, *, freq_bond_type, bt2idx, **kwargs):
    """
    计算边类型的JS距离
    """
    if len(mol_list) == 0:
        return np.nan

    if freq_bond_type[bt2idx['NONE']] != 0:
        freq_bond_type = np.array(freq_bond_type)
        freq_bond_type[bt2idx['NONE']] = 0
        freq_bond_type /= freq_bond_type.sum()

    bond_type_distribution = np.zeros_like(freq_bond_type, dtype=float)

    for mol in mol_list:
        for bond in mol.GetBonds():
            bond_type = str(bond.GetBondType())
            if bond_type in bt2idx:
                bond_type_distribution[bt2idx[bond_type]] += 1

    bond_type_distribution /= bond_type_distribution.sum()

    js_dist = distance.jensenshannon(bond_type_distribution, freq_bond_type)

    return js_dist


def get_charge_type_jsd(mol_list, *, freq_atom_charge, ac2idx, **kwargs):
    """
    计算边缘电荷类型的JS距离
    """
    if len(mol_list) == 0:
        return np.nan

    charge_type_distribution = np.zeros_like(freq_atom_charge, dtype=float)

    for mol in mol_list:
        for atom in mol.GetAtoms():
            charge = atom.GetFormalCharge()

            if charge in ac2idx:
                charge_type_distribution[ac2idx[charge]] += 1

    charge_type_distribution /= charge_type_distribution.sum()

    js_dist = distance.jensenshannon(charge_type_distribution, freq_atom_charge)

    return js_dist


def __calculate_simplicity(molecule):
    num_bonds = molecule.GetNumBonds()
    num_atoms = molecule.GetNumAtoms()

    pred_num_bonds = num_atoms * 1.105 - 0.4899  # coefficients calculated from fitting >1m molecules from ChEMBL

    return abs(num_bonds - pred_num_bonds)


def get_simplicity(mol_list, *args, **kwargs):
    """
    简单性（自己编的），基于用ChEMBL数据拟合出来的n_atom与n_bond（不考虑隐性C-H）线性关系
    """
    if len(mol_list) == 0:
        return np.nan

    values = []
    for mol in mol_list:
        value = __calculate_simplicity(mol)
        values.append(value)

    return np.mean(values)


def calculate_normed_emd(d1, d2):
    """
    The value range is in [0,1], not affected by the number of bins.
    """

    emd = np.abs(d1.cumsum() - d2.cumsum()).mean()
    return emd


def get_nv_emd(mol_list, *, freq_nv, **kwargs):
    """
    计算节点个数的emd
    """
    if len(mol_list) == 0:
        return np.nan

    num_vertices_distribution = np.zeros_like(freq_nv, dtype=float)

    for mol in mol_list:
        num_vertices_distribution[mol.GetNumAtoms()] += 1

    num_vertices_distribution /= num_vertices_distribution.sum()

    emd = calculate_normed_emd(num_vertices_distribution, freq_nv)

    return emd


def get_bond_length_emd(mol_list, *, bt2idx, freq_bond_type, freq_bond_length, bond_length_bins, **kwargs):
    """
    bond_length_bins: [a_0,a_1,...,a_{n-2},a_{n-1}]. a_{n-1}=+inf. If x<a_i, x belongs to i-th bin.
    """
    if len(mol_list) == 0:
        return np.nan

    generated_bond_lengths = {i: np.zeros_like(bond_length_bins) for i in bt2idx.keys() if i != 'NONE'}

    for mol in mol_list:
        conformer = mol.GetConformer()
        atom_pos = np.array(conformer.GetPositions())

        cdists = distance.cdist(atom_pos, atom_pos)

        for i in range(len(atom_pos)):
            for j in range(i):
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is None:
                    # bond_type = 'NONE'
                    continue
                else:
                    bond_type = str(bond.GetBondType())

                generated_bond_lengths[bond_type][np.searchsorted(bond_length_bins, cdists[i, j])] += 1

    for bond_type in generated_bond_lengths.keys():
        s = generated_bond_lengths[bond_type].sum()
        if s != 0:
            generated_bond_lengths[bond_type] /= s
        else:
            generated_bond_lengths[bond_type] += 1 / len(generated_bond_lengths[bond_type])

    emd = 0.
    for bond_type in generated_bond_lengths.keys():
        freq = freq_bond_type[bt2idx[bond_type]]
        if freq > 0:
            emd += freq * calculate_normed_emd(generated_bond_lengths[bond_type],
                                               freq_bond_length[bond_type])

    return emd


def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = 1 - distance.cosine(v1, v2)  # distance.cosine(v1, v2) = 1-cos(v1,v2)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def get_angle_emd(mol_list, *, freq_angle, angle_bins, **kwargs):
    """
    modified from MiDi https://github.com/cvignac/MiDi

    angle_bins: [a_0,a_1,...,a_{n-2},a_{n-1}]. a_{n-1}=+inf. If x<a_i, x belongs to i-th bin. (in degree)
    """
    if len(mol_list) == 0:
        return np.nan

    generated_angle = np.zeros_like(freq_angle)

    for mol in mol_list:
        conformer = mol.GetConformer()  # =mol.GetConformers()[-1]
        atom_pos = np.array(conformer.GetPositions())

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            for i in range(len(neighbors)):
                for j in range(i):
                    angle = calculate_angle(atom_pos[neighbors[i]],
                                            atom_pos[idx],
                                            atom_pos[neighbors[j]])

                    generated_angle[np.searchsorted(angle_bins, angle)] += 1

    generated_angle /= generated_angle.sum()

    emd = calculate_normed_emd(generated_angle, freq_angle)

    return emd


def test_fns():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np

    # 准备分子
    mol_list = [Chem.MolFromSmiles('CC(O)C'), Chem.MolFromSmiles('Cc1ccccc1C=O')]
    mol_list = [Chem.AddHs(i) for i in mol_list]
    AllChem.EmbedMolecule(mol_list[0], AllChem.ETKDG())
    AllChem.EmbedMolecule(mol_list[1], AllChem.ETKDG())

    freq_atom_type = np.array([0.6, 0.2, 0.2])
    at2idx = {'C': 0, 'O': 1, 'H': 2}

    freq_bond_type = np.array([0.7, 0.2, 0.1,2])
    bt2idx = {
        'SINGLE': 0,
        'DOUBLE': 1,
        'AROMATIC': 2,
        'NONE': 3
    }

    freq_atom_charge = np.array([0.9, 0.1])
    ac2idx = {0: 0, 1: 1}  # 电荷类型索引

    freq_nv = np.array(range(100))
    freq_nv = freq_nv / freq_nv.sum()

    freq_bond_length = {
        'SINGLE': np.array([0.5, 0.5]),
        'DOUBLE': np.array([0.5, 0.5]),
        'NONE': np.array([0.5, 0.5]),
        'AROMATIC': np.array([0.5, 0.5])
    }
    bond_length_bins = [1.4, float('inf')]  # 键长分组

    freq_angle = np.array([0.6, 0.4])
    angle_bins = [90, 180]  # 角度分组

    def test_get_atom_type_js():
        js_dist = get_atom_type_jsd(mol_list, freq_atom_type=freq_atom_type, at2idx=at2idx)
        print("JS Distance for Atom Types:", js_dist)
        assert js_dist >= 0, "Jensen-Shannon distance should be non-negative"
        assert js_dist <= 1, "Jensen-Shannon distance should not exceed 1"

    test_get_atom_type_js()

    def test_get_bond_type_js():
        js_dist = get_bond_type_jsd(mol_list, freq_bond_type=freq_bond_type, bt2idx=bt2idx)
        print("JS Distance for Bond Types:", js_dist)
        assert js_dist >= 0, "Jensen-Shannon distance should be non-negative"
        assert js_dist <= 1, "Jensen-Shannon distance should not exceed 1"

    test_get_bond_type_js()

    def test_get_charge_type_js():
        js_dist = get_charge_type_jsd(mol_list, freq_atom_charge=freq_atom_charge, ac2idx=ac2idx)
        print("JS Distance for Charge Types:", js_dist)
        assert js_dist >= 0, "Jensen-Shannon distance should be non-negative"
        assert js_dist <= 1, "Jensen-Shannon distance should not exceed 1"

    test_get_charge_type_js()

    def test_get_simplicity():
        simplicity = get_simplicity(mol_list)
        print("Average simplicity:", simplicity)
        assert simplicity >= 0, "Simplicity should be non-negative"

    test_get_simplicity()

    def test_get_nv_emd():
        wasserstein_dist = get_nv_emd(mol_list, freq_nv=freq_nv)
        print("Wasserstein Distance for number of vertices:", wasserstein_dist)
        assert wasserstein_dist >= 0

    test_get_nv_emd()

    def test_get_bond_length_emd():
        wasserstein_dist = get_bond_length_emd(mol_list, bt2idx=bt2idx, freq_bond_type=freq_bond_type,
                                               freq_bond_length=freq_bond_length, bond_length_bins=bond_length_bins)
        print("Wasserstein Distance for Bond Lengths:", wasserstein_dist)
        assert wasserstein_dist >= 0

    test_get_bond_length_emd()

    def test_get_angle_emd():
        wasserstein_dist = get_angle_emd(mol_list, freq_angle=freq_angle, angle_bins=angle_bins)
        print("Wasserstein Distance for Bond Angles:", wasserstein_dist)
        assert wasserstein_dist >= 0

    test_get_angle_emd()


if __name__ == '__main__':
    test_fns()
