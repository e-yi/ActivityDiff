# modified from MiDi
import torch
import torch.nn as nn

import dgl
import dgl.function as dglfn

from lcmg.utils.g_utils import x2g2x_sub_mean


class PosMLP(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(1, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, g, fv_pos):
        """
        MiDi会在缩放后，将每个图拉至0均值，但是由于padding点的存在，导致实际点不是0均值  todo 这是期望的吗？
        我处理成0均值, 但是看起来这个东西就是SE3Norm，一点用都没有
        """
        norm = torch.norm(fv_pos, dim=-1, keepdim=True)  # bs, n, 1
        new_norm = self.mlp(norm)  # bs, n, 1
        fv_pos_new = fv_pos * new_norm / (norm + self.eps)

        fv_pos_out = x2g2x_sub_mean(g, fv_pos_new, on_node=True)

        return fv_pos_out


class V2GLayer(nn.Module):
    def __init__(self, dv, dg):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(3 * dv, dg)

    def forward(self, g, fv):
        with g.local_scope():
            g.ndata['fv'] = fv
            z = torch.cat([
                dgl.readout_nodes(g, 'fv', op='mean'),
                dgl.readout_nodes(g, 'fv', op='min'),
                dgl.readout_nodes(g, 'fv', op='max'),
                # ! std ignored
            ], dim=1)
            out = self.lin(z)
            return out


class E2GLayer(nn.Module):
    def __init__(self, de, dg):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(3 * de, dg)

    def forward(self, g: dgl.DGLGraph, fe):
        with g.local_scope():
            g.edata['fe'] = fe
            z = torch.cat([
                dgl.readout_edges(g, 'fe', op='mean'),
                dgl.readout_edges(g, 'fe', op='min'),
                dgl.readout_edges(g, 'fe', op='max'),
                # ! std ignored
            ], dim=1)
            out = self.lin(z)
            return out


class E2VLayer(nn.Module):
    def __init__(self, de, dx):
        super().__init__()
        self.lin = nn.Linear(3 * de, dx)

    def forward(self, g: dgl.DGLGraph, fe):
        def reduce_fn(nodes):
            fe2v_dst = nodes.mailbox['fe2v_dst']  # batch_size, de
            return {'me': fe2v_dst.mean(1),
                    'mi': fe2v_dst.min(1).values,
                    'ma': fe2v_dst.max(1).values, }
            # 'std': fe2v_dst.std(1)}  # might cause nan if one node is only connected to a single edge

        with g.local_scope():
            g.edata['fe'] = fe

            # 使用update_all方法进行聚合
            g.update_all(message_func=dglfn.copy_e('fe', 'fe2v_dst'),
                         reduce_func=reduce_fn)

            # z = torch.cat([g.ndata[i] for i in ('me', 'mi', 'ma', 'std')], dim=1)
            z = torch.cat([g.ndata[i] for i in ('me', 'mi', 'ma',)], dim=1)
            out = self.lin(z)
            return out
