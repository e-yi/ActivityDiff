import dgl
import torch
import torch.nn as nn


def _readout_edge_norm(g: dgl.DGLGraph, fv_pos):
    """
    through `_readout_edge_norm()` looks reasonable at first glance,
    one should note that a molecule is represented as a fully-connected graph
    """

    fv2e_pos_diff = dgl.ops.gsddmm(g, 'sub', fv_pos, fv_pos)
    norm = torch.linalg.vector_norm(fv2e_pos_diff, ord=2, dim=-1, keepdim=True)
    mean_norm = dgl.ops.segment_reduce(g.batch_num_edges(), norm, 'mean')
    return mean_norm


def _readout_node_norm(g: dgl.DGLGraph, fv_pos):
    norm = torch.linalg.vector_norm(fv_pos, ord=2, dim=-1, keepdim=True)
    mean_norm = dgl.ops.segment_reduce(g.batch_num_nodes(), norm, 'mean')
    return mean_norm


class PosScaleNormLayer(nn.Module):
    def __init__(self, norm_type='node') -> None:
        super().__init__()

        assert norm_type in ('edge', 'node')
        self.norm_readout_fn = _readout_edge_norm if norm_type == 'edge' else _readout_node_norm

        self.weight = nn.Parameter(torch.empty((1,)))
        self.eps = 1e-8

        nn.init.ones_(self.weight)

    def forward(self, g: dgl.DGLGraph, fv_pos):
        mean_norm = self.norm_readout_fn(g, fv_pos).clamp(min=self.eps)
        mean_norm = dgl.broadcast_nodes(g, mean_norm)
        new_fv = self.weight * fv_pos / mean_norm
        return new_fv


class SE3PointNormLayer(nn.Module):

    def __init__(self, nonlinearity_init_fn=nn.ReLU) -> None:
        super().__init__()

        self.nonlinearity = nonlinearity_init_fn()

        self.layer_norm = nn.LayerNorm(1)
        self.eps = 1e-8

    def forward(self, g: dgl.DGLGraph, fv_pos):
        norm = torch.linalg.vector_norm(fv_pos, ord=2, dim=-1, keepdim=True).clamp(min=self.eps)
        new_norm = self.nonlinearity(self.layer_norm(norm))
        new_fv = new_norm * fv_pos / norm
        return new_fv


class AdaPosScaleNormLayer(nn.Module):

    def __init__(self, dim_g, act_layer, norm_type='node') -> None:
        super().__init__()

        assert norm_type in ('edge', 'node')
        self.norm_readout_fn = _readout_edge_norm if norm_type == 'edge' else _readout_node_norm

        self.linear = nn.Sequential(
            act_layer(),
            nn.Linear(dim_g, 1)
        )

        self.eps = 1e-8

        torch.nn.init.zeros_(self.linear[-1].weight)
        torch.nn.init.zeros_(self.linear[-1].bias)

    def forward(self, g: dgl.DGLGraph, fv_pos, fg):
        mean_norm = self.norm_readout_fn(g, fv_pos).clamp(min=self.eps)
        mean_norm = dgl.broadcast_nodes(g, mean_norm)

        scale = dgl.broadcast_nodes(g, self.linear(fg))

        new_fv_pos = (1 + scale) * fv_pos / mean_norm  ######

        return new_fv_pos
