import pickle
from typing import Dict, Iterable, Optional, Union

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lcmg.runtime_utils.logging_utils import pylogger
from .base_dataset import BaseDataset
from .utils.mask import none_mask, random_mask, random_mask_fragments, random_mask_nodes
from .utils.mol2graph import load_graph, mol2graph

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def bind_lig_pkt_graph(g_lig, g_pkt):
    assert g_lig.batch_size == 1 and g_pkt.batch_size == 1
    # do I really need every possible edges between each graph?
    inter_edge_u, inter_edge_v = torch.meshgrid(torch.arange(g_lig.num_nodes()),
                                                torch.arange(
                                                    g_pkt.num_nodes()),
                                                indexing='ij')
    inter_edge_u = inter_edge_u.reshape(-1).to(g_lig.device)
    inter_edge_v = inter_edge_v.reshape(-1).to(g_lig.device)

    g_full = dgl.heterograph(data_dict={
        ('lig', 'with', 'lig'): g_lig.edges(),
        ('pkt', 'with', 'pkt'): g_pkt.edges(),
        ('lig', 'with', 'pkt'): (inter_edge_u, inter_edge_v),
    },
        num_nodes_dict={'lig': g_lig.num_nodes(), 'pkt': g_pkt.num_nodes()}, idtype=torch.int32)

    g_full, ntype_count, etype_count = dgl.to_homogeneous(
        g_full, return_count=True)

    return g_full, ntype_count, etype_count


def batch_binded_graphs(g_full_list, ntype_count_list, etype_count_list):
    """
    目的是得到一组图，支持（1）batch计算（2）可以使用索引的方式从批量化的子类型图的特征中重整特征以支持1
    与具体特征无关，与边类型、节点类型有关

    具体来说接下来的代码输出的nid使得我可以使用
        torch.cat([feat[i] for i in sorted_ntypes])[nid]
    的方式来得到与dgl.batch([dgl.to_homogeneous(i) for i in hetero_graphs])得到结果匹配的特征

    :param g_full_list:
    :param ntype_count_list:
    :param etype_count_list:
    :return:
    """

    g_full = dgl.batch(g_full_list)

    # https://github.com/dmlc/dgl/blob/191681d8510173576286765237ed0ce87d690e14/python/dgl/heterograph_index.py#L1239C14-L1239C34
    # I will definitely get rid of dgl at some time
    ntypes = sorted(['pkt', 'lig'])
    etypes = sorted(
        [('lig', 'with', 'lig'), ('pkt', 'with', 'pkt'), ('lig', 'with', 'pkt')])

    # node
    node_offsets = [np.cumsum([0] + [ntype_count_i[i] for ntype_count_i in ntype_count_list])
                    for i in range(len(ntypes))]
    for i in range(1, len(node_offsets)):
        node_offsets[i] += node_offsets[i - 1][-1]

    node_counts = [torch.tensor(
        [ntype_count_i[i] for ntype_count_i in ntype_count_list]) for i in range(len(ntypes))]

    for i in range(len(node_offsets)):
        g_full.ndata['_ID'][g_full.ndata['_TYPE'] == i] += (
            torch.repeat_interleave(torch.tensor(node_offsets[i][:-1]), node_counts[i], dim=0)).to(g_full.device)

    # edge
    edge_offsets = [np.cumsum([0] + [etype_count_i[i] for etype_count_i in etype_count_list])
                    for i in range(len(etypes))]
    for i in range(1, len(edge_offsets)):
        edge_offsets[i] += edge_offsets[i - 1][-1]

    edge_counts = [torch.tensor([etype_count_i[i] for etype_count_i in etype_count_list])
                   for i in range(len(etypes))]

    for i in range(len(edge_offsets)):
        g_full.edata['_ID'][g_full.edata['_TYPE'] == i] += (
            torch.repeat_interleave(torch.tensor(edge_offsets[i][:-1]), edge_counts[i], dim=0)).to(g_full.device)

    fg_features = {'eid': g_full.edata.pop('_ID'), 'nid': g_full.ndata.pop('_ID'),
                   'ntypes': ntypes, 'etypes': etypes}

    return g_full, fg_features


class LCMGDataset(BaseDataset):
    """
    todo
        store features in DGLGraph add `gdata` to store graph level data
        support mask
        support other conditions in the future
        可以加别的特征，只需要能够对加噪后的分子也计算一份特征即可
    """

    def __init__(self,
                 mol_pkl_path,
                 info: Optional[Dict] = None,
                 info_pkl_path=None,

                 label=None,
                 label_path=None,

                 # options
                 mask=True,  # use mask
                 fix_CoM=False,
                 wo_conformer=False,
                 ):
        """

        todo: add random rotation to lig and conditions when pkt or phar exists
        Args:
            dgl_graphs: fully-connected graphs
            dgl_graph_path: [recommended] use path so that data is not saved in checkpoints
            info: [recommended] directly input `info` so that it is saved in checkpoints
            info_pkl_path:
            fix_CoM: whether to move center of mass (CoM) to a fixed point, can be ignored when use aligned loss

        """
        super().__init__()

        self.mask = mask
        if fix_CoM:
            # should move the CoM of all conditional points to 0
            raise NotImplementedError(f'{fix_CoM=}')

        self.wo_conformer = wo_conformer

        self.mols = pd.read_pickle(mol_pkl_path)

        if info is None:
            info = dict(pd.read_pickle(info_pkl_path))
        self.info = info

        self._len = len(self.mols)

        self.info.update(
            {
                'nv_atom_type': len(info['at2idx']),
                'nv_atom_charge': len(info['ac2idx']),
                'nv_pos_dim': info['n_pos_dim'],
                'ne_bond_type': len(info['bt2idx'])
            }
        )

        self.label = label
        if self.label is None and label_path is not None:
            self.label = pd.read_pickle(label_path)

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        mol = self.mols[item]
        g_lig = mol2graph(mol, cfg=self.info, as_complete_graph=True,
                          fill_g_data=True, wo_conformer=self.wo_conformer)

        masks = {}
        if not self.wo_conformer:
            masks |= none_mask(g_lig, ('v_pos',))

        if self.mask:
            # todo tune the mask scheme
            x = np.random.rand()
            match x:
                case x if x < 0.2:
                    masks |= random_mask(
                        g_lig, 0.3, ('v_atom_type', 'v_atom_charge', 'e_bond_type'))
                case x if x < 0.4:
                    masks |= random_mask_nodes(
                        g_lig, 0.2, ('v_atom_type', 'v_atom_charge', 'e_bond_type'))
                case x if x < 0.6:
                    try:
                        masks |= random_mask_fragments(
                            g_lig, mol, ('v_atom_type', 'v_atom_charge', 'e_bond_type'))
                    except ValueError:
                        masks |= random_mask_nodes(
                            g_lig, 0.5, ('v_atom_type', 'v_atom_charge', 'e_bond_type'))
                case _:
                    masks |= none_mask(
                        g_lig, ('v_atom_type', 'v_atom_charge', 'e_bond_type'))
        else:
            masks |= none_mask(
                g_lig, ('v_atom_type', 'v_atom_charge', 'e_bond_type'))

        for k, v in masks.items():
            data = g_lig.ndata if k.startswith('v') else g_lig.edata
            data[f'm{k}'] = v

        if self.label is not None:
            return g_lig, self.label[item]

        return g_lig

    def collate_fn(self, batch):
        if self.label is not None:
            g_lig, labels = list(zip(*batch))
            g_lig = dgl.batch(g_lig)
            labels = torch.FloatTensor(labels)
        else:
            g_lig = dgl.batch(batch)

        for prop in ('v_atom_type', 'v_atom_charge', 'e_bond_type'):
            data = g_lig.ndata if prop.startswith('v') else g_lig.edata
            data[f'f{prop}'] = F.one_hot(
                data[f'f{prop}'], self.info[f'n{prop}']).float()

        lig_features = {}
        lig_features.update({i: g_lig.ndata.pop(i)
                            for i in list(g_lig.ndata.keys())})
        lig_features.update({i: g_lig.edata.pop(i)
                            for i in list(g_lig.edata.keys())})

        res = {
            'lig': {
                'graph': g_lig,
                'feature': lig_features,
            },
        }

        if self.label is not None:
            res['label'] = labels

        return res
