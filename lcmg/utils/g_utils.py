import dgl
import torch


def get_graph_node_num(g: dgl.DGLGraph):
    return g.num_nodes()


def get_graph_edge_num(g: dgl.DGLGraph):
    return g.num_edges()


def quaternion_to_rotation_matrix(quaternion):
    a, b, c, d = torch.unbind(quaternion, -1)

    r = torch.stack(
        (
            a * a + b * b - c * c - d * d,
            2 * b * c - 2 * a * d,
            2 * b * d + 2 * a * c,
            2 * b * c + 2 * a * d,
            a * a - b * b + c * c - d * d,
            2 * c * d - 2 * a * b,
            2 * b * d - 2 * a * c,
            2 * c * d + 2 * a * b,
            a * a - b * b - c * c + d * d,
        ),
        -1,
    ).reshape(quaternion.shape[:-1] + (3, 3))

    return r


@torch.no_grad()
def get_align_r_t(pos, ref_pos, weights=None, graph: dgl.DGLGraph = None):
    # adapted from https://github.com/DirectMolecularConfGen/DMCG/blob/23e90f7231eac9a065093ee44b378cbf65181a7c/confgen/model/gnn.py#L588

    if graph is None:
        graph = dgl.rand_graph(pos.size(0), 0).to(pos.device)

    if weights is None:
        weights = torch.ones((pos.shape[0],), device=pos.device, dtype=torch.float)

    weights = weights.unsqueeze(1)

    weights_mean = dgl.ops.segment_reduce(graph.batch_num_nodes(), weights, reducer='mean')
    pos_x_mean = dgl.ops.segment_reduce(graph.batch_num_nodes(), pos * weights, reducer='mean') / weights_mean
    pos_y_mean = dgl.ops.segment_reduce(graph.batch_num_nodes(), ref_pos * weights, reducer='mean') / weights_mean

    total_nodes = ref_pos.shape[0]
    num_graphs = graph.batch_size

    x = pos - dgl.broadcast_nodes(graph, pos_x_mean)
    y = ref_pos - dgl.broadcast_nodes(graph, pos_y_mean)
    a = y + x
    b = y - x

    a = a.view(total_nodes, 1, 3)
    b = b.view(total_nodes, 3, 1)
    tmp0 = torch.cat(
        [b.new_zeros((1, 1, 1)).expand(total_nodes, -1, -1), -b.permute(0, 2, 1)], dim=-1
    )
    eye = torch.eye(3).to(a).unsqueeze(0).expand(total_nodes, -1, -1)
    a = a.expand(-1, 3, -1)
    tmp1 = torch.cross(eye, a, dim=-1)
    tmp1 = torch.cat([b, tmp1], dim=-1)
    tmp = torch.cat([tmp0, tmp1], dim=1)  # n,4,4
    tmpb = torch.bmm(tmp.permute(0, 2, 1), tmp).view(total_nodes, -1)

    tmpb = tmpb * weights  ########

    tmpb = dgl.ops.segment_reduce(graph.batch_num_nodes(), tmpb, reducer='mean').view(num_graphs, 4, 4)

    w, v = torch.linalg.eigh(tmpb)
    # min_rmsd = w[:, 0]
    min_q = v[:, :, 0]
    rotation = quaternion_to_rotation_matrix(min_q)
    t = pos_y_mean - torch.einsum("kj,kij->ki", pos_x_mean, rotation)

    return rotation, t


def apply_r_t(graph, pos, r, t):
    r = dgl.broadcast_nodes(graph, r)
    t = dgl.broadcast_nodes(graph, t)

    pos = torch.einsum('bj,bij->bi', pos, r) + t

    return pos


def x2g_readout(g, x, on_node, op):
    with g.local_scope():
        if on_node:
            g.ndata['x'] = x
            readout_fn = dgl.readout_nodes
        else:
            g.edata['x'] = x
            readout_fn = dgl.readout_edges

        fx2g_op = readout_fn(g, 'x', op=op)

        return fx2g_op


def x2g_mean(g, x, on_node):
    return x2g_readout(g, x, on_node, 'mean')


def x2g_sum(g, x, on_node):
    return x2g_readout(g, x, on_node, 'sum')


def g2x_copy(g, x, on_node):
    broadcast_fn = dgl.broadcast_nodes if on_node else dgl.broadcast_edges

    fg2x_copy = broadcast_fn(g, x)

    return fg2x_copy


def x2g2x_sub_mean(g, x, on_node):
    fx2g_mean = x2g_mean(g, x, on_node)
    fx_mean = g2x_copy(g, fx2g_mean, on_node)
    fx_zero_centered = x - fx_mean

    return fx_zero_centered
