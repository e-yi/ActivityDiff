from collections import defaultdict

import numpy as np
from rdkit import Chem

from lcmg.evaluations.distribution_distances import get_angle_emd, get_atom_type_jsd, get_bond_length_emd, \
    get_bond_type_jsd, \
    get_charge_type_jsd, get_nv_emd, get_simplicity

from lcmg.runtime_utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# each functions returns a float or a dict[str, float]
ALL_SAMPLING_METRICS_FN = {
    'connectivity': None,
    'validity': None,
    'uniqueness': None,
    'novelty': None,

    'atom_type_jsd': get_atom_type_jsd,
    'bond_type_jsd': get_bond_type_jsd,
    'charge_type_jsd': get_charge_type_jsd,
    'bond_length_emd': get_bond_length_emd,
    'angle_emd': get_angle_emd,
    'nv_emd': get_nv_emd,

    'simplicity': get_simplicity,
}


def get_sampling_metrics(mol_list, ref_info, train_smiles=None, metrics='all', kekulize=False):
    """
    Args:
        mol_list (list): mols, only the first comformers is considered,
            todo only the connected molecules is considered? or the largest frag?

    """
    if metrics == 'all':
        metrics = ALL_SAMPLING_METRICS_FN
    else:
        assert isinstance(metrics, list)
        metrics = {i: ALL_SAMPLING_METRICS_FN[i] for i in metrics}

    res = {}

    error_message = defaultdict(int)

    valid = []

    valid_mol_list = []
    for mol in mol_list:

        if mol is not None:
            try:
                mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                if len(mol_frags) > 1:
                    error_message['NotConnected'] += 1
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                Chem.SanitizeMol(largest_mol)
                if kekulize:
                    Chem.Kekulize(largest_mol, clearAromaticFlags=True)
                smiles = Chem.MolToSmiles(largest_mol)
                valid.append(smiles)
                valid_mol_list.append(largest_mol)
            except Chem.rdchem.AtomValenceException:
                error_message['AtomValenceException'] += 1
            except Chem.rdchem.KekulizeException:
                error_message['KekulizeException'] += 1
            except (Chem.rdchem.AtomKekulizeException, ValueError):
                error_message['AtomKekulizeException'] += 1

    log.warning(error_message)

    res['smiles'] = list(valid)

    connectivity = 1 - (error_message['NotConnected'] / len(mol_list))
    validity = len(valid) / len(mol_list)

    unique = set(valid)
    uniqueness = len(unique) / len(valid) if len(valid) != 0 else np.nan

    res['connectivity'] = connectivity
    res['validity'] = validity
    res['uniqueness'] = uniqueness

    if 'novelty' in metrics and train_smiles is not None:
        if not isinstance(train_smiles, set):
            train_smiles = set(train_smiles)
        novelty = len(unique - train_smiles) / len(unique) if len(unique) != 0 else np.nan
        res['novelty'] = novelty

    for k, fn in metrics.items():
        if k in ['connectivity', 'validity', 'uniqueness', 'novelty']:
            continue

        m = fn(valid_mol_list, **ref_info)
        if isinstance(m, dict):
            for mk, mv in m.items():
                res[f'{k}_{mk}'] = mv
        else:
            res[k] = m

    return res


# In[212]:

def test_fn():
    from rdkit.Chem import AllChem

    # 准备分子数据
    smiles_list = ['CC', 'CC.C', 'O=C(O)C', 'O=C(O)C', 'C1=CC=CC=C1', '[Na]']  # 包含无效和特殊情况的SMILES
    mol_list = [mol for smiles in smiles_list
                if (mol := Chem.MolFromSmiles(smiles)) is not None]
    for m in mol_list:
        AllChem.EmbedMolecule(m)

    # 创建参考信息，这些通常是分子属性的预期分布
    ref_info = {
        'freq_atom_type': np.array([0.6, 0.19, 0.01, 0.2]),
        'at2idx': {'C': 0, 'O': 1, 'Na': 2, 'H': 3},
        'freq_bond_type': np.array([0.7, 0.2, 0.1]),
        'bt2idx': {'SINGLE': 0, 'DOUBLE': 1, 'AROMATIC': 2},
        'freq_atom_charge': np.array([0.9, 0.1]),
        'ac2idx': {0: 0, 1: 1},
        'freq_bond_length': {
            'SINGLE': np.array([0.5, 0.5]),
            'DOUBLE': np.array([0.5, 0.5]),
            'AROMATIC': np.array([0.5, 0.5]),
        },
        'bond_length_bins': [1.4, float('inf')],
        'freq_angle': np.array([0.6, 0.4]),
        'angle_bins': [90, 180],
        'freq_nv': np.array([0.02]*50),
    }

    train_smiles = ['CC', 'CCC', 'CCCC']  # 训练集中的SMILES

    results = get_sampling_metrics(mol_list, ref_info, train_smiles, metrics='all')

    # 测试基本的度量是否正确
    assert 0 <= results['connectivity'] <= 1, "Connectivity should be between 0 and 1"
    assert 0 <= results['validity'] <= 1, "Validity should be between 0 and 1"
    assert 0 <= results['uniqueness'] <= 1, "Uniqueness should be between 0 and 1"

    # 如果提供了train_smiles，测试新颖性
    if 'novelty' in results:
        assert 0 <= results['novelty'] <= 1, "Novelty should be between 0 and 1"

    # 测试其他度量
    for key in ['atom_type_jsd', 'bond_type_jsd', 'charge_type_jsd', 'bond_length_emd', 'angle_emd', 'simplicity']:
        assert key in results, f"{key} should be calculated and included in results"
        assert isinstance(results[key], float), f"{key} should be a float value"

    print(results)

    print("All tests passed.")


if __name__ == "__main__":
    test_fn()
