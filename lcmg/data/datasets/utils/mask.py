"""
mask=1 means to keep this feature!
"""

import dgl
import torch

from lcmg.data.datasets.utils.constants import Properties
from lcmg.utils.mol_utils import random_break_brics_bonds


# key is like `v_atom_type`

def all_mask(g: dgl.DGLGraph, properties):
    assert len(set(properties) - Properties.ANY) == 0
    masks = {}
    for prop in properties:
        data = g.ndata if prop.startswith('v') else g.edata
        feature = data[f'f{prop}']
        masks[prop] = feature.new_ones((feature.shape[0],)).bool()
    return masks


def none_mask(g: dgl.DGLGraph, properties):
    assert len(set(properties) - Properties.ANY) == 0
    masks = {}
    for prop in properties:
        data = g.ndata if prop.startswith('v') else g.edata
        feature = data[f'f{prop}']
        masks[prop] = feature.new_zeros((feature.shape[0],)).bool()
    return masks


def random_mask(g: dgl.DGLGraph, p, properties):
    assert len(set(properties) - Properties.ANY) == 0
    masks = {}
    for prop in properties:
        size = g.num_nodes() if prop.startswith('v') else g.num_edges()
        mask = torch.bernoulli(torch.full((size,), p, dtype=torch.float32, device=g.device))
        masks[prop] = mask.bool()

    return masks


def mask_nodes(g, node_mask, properties):
    # 初始化节点和边的掩码
    node_masks = {f'{prop}': torch.zeros((g.num_nodes(),), dtype=torch.bool) for prop in properties if
                  prop.startswith('v')}
    edge_masks = {f'{prop}': torch.zeros((g.num_edges(),), dtype=torch.bool) for prop in properties if
                  prop.startswith('e')}

    # 为被选中的节点设置掩码
    for prop in node_masks:
        node_masks[prop] = node_mask.clone()

    # 获取与被选中节点相连的所有边的ID
    nodes_to_mask = torch.nonzero(node_mask, as_tuple=True)[0].int()
    in_edges_to_mask = g.in_edges(nodes_to_mask, form='eid')
    out_edges_to_mask = g.out_edges(nodes_to_mask, form='eid')
    edges_to_mask = torch.cat([in_edges_to_mask, out_edges_to_mask], dim=0).int()

    # 为这些边设置掩码
    for prop in edge_masks:
        edge_masks[prop][edges_to_mask] = 1

    # 合并节点和边的掩码到一个字典中
    masks = {**node_masks, **edge_masks}

    return masks


def random_mask_nodes(g: dgl.DGLGraph, p: float, properties):
    assert len(set(properties) - Properties.ANY) == 0
    node_mask = torch.bernoulli(torch.full((g.num_nodes(),), p, dtype=torch.float32, device=g.device))
    node_mask = node_mask.bool()

    return mask_nodes(g, node_mask, properties)


def mask_fragments(g, node_mask, properties, anchor_idx=None):
    """
    Noted that this would also mask(fix) the connection bonds (single in BRICS)
        between the given fragment and the rest parts.
    :param g:
    :param node_mask:
    :param properties:
    :param anchor_idx: not used
    :return:
    """
    masks = mask_nodes(g, node_mask, properties)

    return masks


def random_mask_fragments(g, mol, properties, n_cut=None, strict_cut=False):
    """
    Randomly masks fragments of a molecule and returns a dictionary of masks.

    This function uses the BRICS algorithm to randomly cut the molecule into fragments
    and generates masks for nodes and edges based on the resulting fragments.

    If `n_cut` is 1, the molecule is cut into 2 fragments.
    If `n_cut` is 2, the molecule is cut into 3 fragments, with the 2 parts at the end being masked,
        corresponding to the linker design problem.

    :param g: DGL graph object representing the molecular graph.
    :param mol: RDKit molecule object to be randomly fragmented.
    :param properties: List of properties to mask. Properties can be node properties (starting with 'v') or edge
                       properties (starting with 'e').
    :param n_cut: Number of cuts to make in the molecule. If None, a random number between 1 and 2 is chosen.
    :param strict_cut: if True and `n_cut` is given, then raise Exception when `mol` cannot be cut into
        desired number of parts,

    :return: Dictionary containing masks for nodes and edges.
    """
    if n_cut is None:
        strict_cut = False
        n_cut = torch.randint(1, 3, (1,)).item()  # 1 or 2

    assert n_cut == 1 or n_cut == 2

    fragment_idx, anchor_atom_idx, fragmented_mol = random_break_brics_bonds(mol, num_bonds_to_cut=n_cut,
                                                                             return_frag_mol=True, strict_cut=strict_cut)
    fragment_idx = [i for j in fragment_idx for i in j]
    anchor_atom_idx = [i for j in anchor_atom_idx for i in j]

    node_mask = torch.zeros((g.num_nodes(),), dtype=torch.bool, device=g.device)
    node_mask[fragment_idx] = True

    return mask_fragments(g, node_mask, properties, anchor_idx=anchor_atom_idx)


def test_fn():
    import torch
    from mol2graph import mol2graph
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles = "CC1=CC=CC=C1CC(=O)NC2CC2"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    mol = Chem.RemoveHs(mol)

    cfg = {'atom_types': 'H,B,C,N,O,F,Al,Si,P,S,Cl,As,Br,I,Hg,Bi,Se,Fe,Pt,V,Rh,Co,Ru,Mg'.split(','),
           'atom_charges': [-2, -1, 0, 1, 2, 3],
           'bond_types': ['NONE', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'], }
    cfg['at2idx'] = {v: i for i, v in enumerate(cfg['atom_types'])}
    cfg['ac2idx'] = {v: i for i, v in enumerate(cfg['atom_charges'])}
    cfg['bt2idx'] = {v: i for i, v in enumerate(cfg['bond_types'])}
    cfg['idx2at'] = {v: k for k, v in cfg['at2idx'].items()}
    cfg['idx2ac'] = {v: k for k, v in cfg['ac2idx'].items()}
    cfg['idx2bt'] = {v: k for k, v in cfg['bt2idx'].items()}

    graph = mol2graph(mol, cfg, as_complete_graph=True, fill_g_data=True)

    uid, vid = graph.edges()

    g = graph

    def apply_mask_to_data(g, masks):
        masked_data = {}
        for key, mask in masks.items():
            prefix = 'v' if 'v' in key else 'e'
            feature_key = 'f' + key
            if prefix == 'v':
                original_data = g.ndata[feature_key]
            else:
                original_data = g.edata[feature_key]

            noisy_data = torch.ones_like(original_data) * -9
            noisy_data[mask] = original_data[mask]
            masked_data[feature_key] = noisy_data
        return masked_data

    def print_data(data):
        for k, v in data.items():
            if k[0] == 'f':
                k = k[1:]
            else:
                v = v.int()

            if k[0] == 'v':
                print(f'{k}\n{v}')
            else:
                x = torch.zeros([graph.num_nodes()] * 2, dtype=v.dtype)
                x[uid, vid] = v
                print(f'{k}\n{x}')

    def test_none_mask():
        print('------------------')
        print('------------------')
        print("none_mask")
        masks = none_mask(g, {'v_atom_type', 'v_atom_charge', 'e_bond_type'})
        assert masks['v_atom_type'].sum() == 0
        assert masks['v_atom_charge'].sum() == 0
        assert masks['e_bond_type'].sum() == 0
        print_data(masks)
        print('------------------')
        apply_mask_to_data(g, masks)
        return g, masks

    def test_random_mask():
        print('------------------')
        print('------------------')
        print("random_mask")
        p = 0.5
        masks = random_mask(g, p, {'v_atom_type', 'v_atom_charge', 'e_bond_type'})
        assert masks['v_atom_type'].size(0) == g.num_nodes()
        assert masks['v_atom_charge'].size(0) == g.num_nodes()
        assert masks['e_bond_type'].size(0) == g.num_edges()
        assert (masks['v_atom_type'] == 0).sum() > 0
        assert (masks['v_atom_charge'] == 0).sum() > 0
        assert (masks['e_bond_type'] == 0).sum() > 0
        print_data(masks)
        print('------------------')
        apply_mask_to_data(g, masks)
        return g, masks

    def test_random_mask_nodes():
        print('------------------')
        print('------------------')
        print("random_mask_nodes")
        p = 0.2
        masks = random_mask_nodes(g, p, {'v_atom_type', 'v_atom_charge', 'e_bond_type', 'v_pos'})
        assert masks['v_atom_type'].size(0) == g.num_nodes()
        assert masks['v_atom_charge'].size(0) == g.num_nodes()
        assert masks['e_bond_type'].size(0) == g.num_edges()
        print_data(masks)
        print('------------------')
        print_data(apply_mask_to_data(g, masks))
        return g, masks

    def test_mask_fragments():
        print('------------------')
        print('------------------')
        print("mask_fragments")

        # linker-like
        node_mask = torch.ones(14).bool()
        node_mask[[7, 8, 9, 10]] = 0

        anchor_idx = [6, 11]
        properties = {'v_atom_type', 'v_atom_charge', 'e_bond_type'}
        masks = mask_fragments(g, node_mask, properties, anchor_idx=anchor_idx)

        print_data(masks)
        print('------------------')
        print_data(apply_mask_to_data(g, masks))
        return g, masks

    def test_random_mask_fragments():
        print("random_mask_fragments")

        masks = random_mask_fragments(g, mol, {'v_atom_type', 'v_atom_charge', 'e_bond_type'}, n_cut=None)

        print_data(masks)
        print('------------------')
        print_data(apply_mask_to_data(g, masks))
        return g, masks

    test_none_mask()
    test_random_mask()
    test_random_mask_nodes()
    test_mask_fragments()
    test_random_mask_fragments()


if __name__ == "__main__":
    test_fn()
