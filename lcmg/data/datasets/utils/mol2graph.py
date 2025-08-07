from typing import Iterable, Union

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from scipy.spatial import distance

from lcmg.data.datasets.utils.constants import Properties
from lcmg.data.datasets.utils.mask import mask_fragments


def save_graph(filename, g_list):
    """

    :param filename:
    :param g_list: a list of DGLGraphs, should not contain any features within each graph
    :return:
    """
    batched_g = dgl.batch(g_list)
    bnn = batched_g.batch_num_nodes()
    bne = batched_g.batch_num_edges()

    dgl.save_graphs(filename, [batched_g], {'batch_num_nodes': bnn, 'batch_num_edges': bne})


def load_graph(filename, as_batched_graph=True):
    batched_g, p = dgl.load_graphs(filename)
    batched_g = batched_g[0]

    batched_g.set_batch_num_nodes(p['batch_num_nodes'])
    batched_g.set_batch_num_edges(p['batch_num_edges'])

    if as_batched_graph:
        return batched_g
    else:
        return dgl.unbatch(batched_g)


def _calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = 1 - distance.cosine(v1, v2)  # distance.cosine(v1, v2) = 1-cos(v1,v2)
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def get_info_from_mol(mol: Chem.Mol, cfg, collect_geo_info=True, count_none_edge=True):
    """
    :param mol:
    :param cfg: must have 'at2idx` `ac2idx` `bt2idx` `bond_length_bins` `angle_bins`
    :param collect_geo_info:
    :param count_none_edge: whether to count NONE edge type, should be true if using complete graph to represent mol
    :return: mol_info
    """

    at2idx = cfg['at2idx']
    ac2idx = cfg['ac2idx']
    bt2idx = cfg['bt2idx']

    mol_info = {
        'count_atom_type': np.zeros((len(at2idx),), dtype=int),
        'count_atom_charge': np.zeros((len(ac2idx),), dtype=int),
        'count_bond_type': np.zeros((len(bt2idx),), dtype=int),
    }

    if collect_geo_info:
        bond_length_bins = cfg['bond_length_bins']
        angle_bins = cfg['angle_bins']

        mol_info.update(
            {
                'count_bond_length': {i: np.zeros_like(bond_length_bins, dtype=int) for i in bt2idx.keys() if
                                      i != 'NONE'},
                'count_angle': np.zeros_like(angle_bins, dtype=int)
            }
        )

    # chemical
    for atom in mol.GetAtoms():
        mol_info['count_atom_type'][at2idx[str(atom.GetSymbol())]] += 1
        mol_info['count_atom_charge'][ac2idx[atom.GetFormalCharge()]] += 1

    n_none_edge = mol.GetNumAtoms()*(mol.GetNumAtoms()-1)//2
    for bond in mol.GetBonds():
        bond_type = str(bond.GetBondType())
        if bond_type == 'DATIVE':
            continue
        mol_info['count_bond_type'][bt2idx[bond_type]] += 1
        n_none_edge -= 1
    if count_none_edge:
        mol_info['count_bond_type'][bt2idx['NONE']] = n_none_edge

    # geometry
    if collect_geo_info:
        conformer = mol.GetConformer()
        atom_pos = np.array(conformer.GetPositions())

        ## bond length
        for bond in mol.GetBonds():
            bond_type = str(bond.GetBondType())
            if bond_type == 'DATIVE':
                continue

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            dist = np.linalg.norm(atom_pos[i] - atom_pos[j], ord=2)

            mol_info['count_bond_length'][bond_type][np.searchsorted(bond_length_bins, dist)] += 1

        ## angle
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            for i in range(len(neighbors)):
                for j in range(i):
                    angle = _calculate_angle(atom_pos[neighbors[i]],
                                             atom_pos[idx],
                                             atom_pos[neighbors[j]])

                    mol_info['count_angle'][np.searchsorted(angle_bins, angle)] += 1

    return mol_info


def mol2graph(mol, cfg: dict, *,
              as_complete_graph=True,
              wo_conformer=False,
              int_dgl_graph=True,
              fill_g_data=False,
              onehot=False,
              ):
    """

    :param mol:
    :param cfg: must have `at2idx` `ac2idx` `bt2idx`
    :param as_complete_graph:
    :return:
    """
    atom_types = []
    atom_charges = []

    for atom in mol.GetAtoms():
        atom_types.append(cfg['at2idx'][str(atom.GetSymbol())])
        atom_charges.append(cfg['ac2idx'][atom.GetFormalCharge()])

    atom_types = torch.LongTensor(atom_types)
    atom_charges = torch.LongTensor(atom_charges)

    if wo_conformer:
        atom_pos = torch.zeros(mol.GetNumAtoms(), 3)
    else:
        atom_pos = mol.GetConformers()[0].GetPositions()
        atom_pos = torch.FloatTensor(atom_pos)

    if as_complete_graph:
        n_atoms = mol.GetNumAtoms()
        non_edge_id = cfg['bt2idx']['NONE']
        half_edge_feature = torch.ones(n_atoms, n_atoms).long() * non_edge_id
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            if u < v:
                u, v = v, u  # so (u,v) is at the lower triangle
            bond_type = str(bond.GetBondType())
            bond_type = 'NONE' if bond_type == 'DATIVE' else bond_type
            half_edge_feature[u, v] = cfg['bt2idx'][bond_type]
        us, vs = torch.tril_indices(n_atoms, n_atoms, offset=-1).unbind()
        half_edge_feature = half_edge_feature[us, vs]

        src = torch.cat((us, vs))
        dst = torch.cat((vs, us))
        edge_feature = torch.cat((half_edge_feature, half_edge_feature))
        bond_types = torch.LongTensor(edge_feature)
    else:
        bond_types = []
        us, vs = [], []
        for bond in mol.GetBonds():
            bond_type = str(bond.GetBondType())
            if bond_type == 'DATIVE':
                continue

            us.append(bond.GetBeginAtomIdx())
            vs.append(bond.GetEndAtomIdx())

            bond_types.append(cfg['bt2idx'][bond_type])

        src, dst = torch.LongTensor(us + vs), torch.LongTensor(vs + us)
        bond_types = torch.LongTensor(bond_types + bond_types)

    graph = dgl.graph((src, dst), num_nodes=mol.GetNumAtoms())
    if onehot:
        atom_types = F.one_hot(atom_types, num_classes=len(cfg['at2idx'])).float()
        atom_charges = F.one_hot(atom_charges, num_classes=len(cfg['ac2idx'])).float()
        bond_types = F.one_hot(bond_types, num_classes=len(cfg['bt2idx'])).float()

    features = {
        'v_atom_type': atom_types,
        'v_atom_charge': atom_charges,
        'v_pos': atom_pos,
        'e_bond_type': bond_types
    }

    if int_dgl_graph:
        graph = graph.int()
    else:
        raise NotImplementedError(f'{int_dgl_graph=}')

    if fill_g_data:
        for k, v in features.items():
            data = graph.ndata if k.startswith('v') else graph.edata
            data[f'f{k}'] = v
        return graph

    return graph, features


def build_complete_graphs(n_nodes: Union[int, Iterable[int]], int_dgl_graph=True):
    if not isinstance(n_nodes, Iterable):
        n_nodes = [n_nodes]

    graphs = []
    for n_nodes_i in n_nodes:
        us, vs = torch.tril_indices(n_nodes_i, n_nodes_i, offset=-1).unbind()
        src = torch.cat((us, vs))
        dst = torch.cat((vs, us))
        graph = dgl.graph((src, dst), num_nodes=n_nodes_i)
        graphs.append(graph)

    graphs = dgl.batch(graphs)

    if int_dgl_graph:
        graphs = graphs.int()

    return graphs


def build_graphs_with_fragment(
        n_nodes: Union[int, Iterable[int]],
        info: dict,
        fragment: Chem.Mol,
        only_link_anchor=True,
        use_conformer=False,
        int_dgl_graph=True,
):
    """
    Build graphs with a given molecular fragment. This function allows for the construction of graphs by adding
    virtual atoms (represented by `*` in the fragment) and optionally linking only anchor atoms.

    :param n_nodes: An integer or an iterable of integers representing the number of nodes in the resulting graphs.
    :param info: A dictionary containing additional information required for graph construction.
    :param fragment: An RDKit `Chem.Mol` object representing the molecular fragment to be used as the basis for graph
                     construction.
    :param only_link_anchor: A boolean indicating whether to link only anchor atoms (represented in SMILES by `*`,
                             e.g., "[*]CO"). If True, only the anchor atoms will be linked.
                             There are two ways to do this. One is to not connect non-anchor atoms in the fragment
                             at all. No edge, no bond. However, this approach conflicts with the current training
                             procedure and might cause the interaction of non-connected atoms to be ignored. Another
                             way is to use mask and fix some edge types to be NONE.
                             todo definitely consider using augmented point cloud like method operating like dMaSIF.
    :param use_conformer: A boolean indicating whether to use conformer information for graph construction. If False,
                          only chemical information will be used.
    :param int_dgl_graph: A boolean indicating whether to convert the resulting DGL graph to an integer type.

    :return: A batch of DGL graphs constructed from the given fragment, with optional anchor atom linkage and conformer
             usage.

    :raises ValueError: If `only_link_anchor` is True and the fragment does not contain any anchor atoms (indicated by `*`).
    """

    if isinstance(n_nodes, int):
        n_nodes = [n_nodes]

    virtual_atom_idx = {atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetSymbol() == '*'}
    if len(virtual_atom_idx) == 0 and only_link_anchor:
        raise ValueError('use `*` to indicate anchor atoms, e.g. [1*]COOH')

    fragment_idx = [atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetSymbol() != '*']
    anchor_atom_idx = [j.GetIdx() for i in virtual_atom_idx for j in fragment.GetAtomWithIdx(i).GetNeighbors()]

    graphs = []
    # 这一段可以提高效率，如果明确确定edge和eid的映射关系，这样就只需跑一次mol2graph，然后把特征映射到完全图的一部分就行
    # 实现很简单，但是暂时不做
    for n_node in n_nodes:
        em = Chem.EditableMol(fragment)

        n_add = max(0, n_node - fragment.GetNumAtoms())  ##

        for v in virtual_atom_idx:
            em.ReplaceAtom(v, Chem.Atom(6))
        for i in range(n_add):
            em.AddAtom(Chem.Atom(6))
        g = mol2graph(em.GetMol(), info,
                      as_complete_graph=True,
                      wo_conformer=not use_conformer,
                      fill_g_data=True,
                      onehot=True)

        node_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
        node_mask[fragment_idx] = True

        properties = Properties.LIGAND if use_conformer else Properties.LIGAND_CHEM
        anchor_idx = anchor_atom_idx if only_link_anchor else None
        masks = mask_fragments(g, node_mask, properties, anchor_idx=anchor_idx)

        for k, v in masks.items():
            data = g.ndata if k.startswith('v') else g.edata
            data[f'm{k}'] = v

        graphs.append(g)

    graphs = dgl.batch(graphs)
    if int_dgl_graph:
        graphs = graphs.int()

    return graphs
