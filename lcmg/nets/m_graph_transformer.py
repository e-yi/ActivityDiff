import dgl
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from lcmg.nets.layers import AdaPosScaleNormLayer
from lcmg.utils.g_utils import x2g2x_sub_mean


class Transition(nn.Module):
    """ af3 Algorithm 11"""

    def __init__(self, dim, n=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear_ab = nn.Linear(dim, dim * n * 2, bias=False)
        self.linear_x = nn.Linear(dim * n, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        a, b = self.linear_ab(x).chunk(2, dim=-1)
        x = self.linear_x(F.silu(a) * b)
        return x


class NormF(nn.Module):

    def __init__(self, dv, de, dg):
        super().__init__()

        self.norm_v = nn.LayerNorm(dv)
        self.norm_e = nn.LayerNorm(de)
        self.norm_g = nn.LayerNorm(dg)
        self.norm_v_pos = nn.Identity()

    def forward(self, g, fv, fe, fg, fv_pos):
        fv = self.norm_v(fv)
        fe = self.norm_e(fe)
        fg = self.norm_g(fg)
        fv_pos = self.norm_v_pos(fv_pos)

        return fv, fe, fg, fv_pos


class AdaLN(nn.Module):

    def __init__(self, dv, de, dg, act_layer):
        super().__init__()

        self.norm_v = nn.LayerNorm(dv, elementwise_affine=False)
        self.norm_e = nn.LayerNorm(de, elementwise_affine=False)
        self.norm_g = nn.LayerNorm(dg)
        self.norm_v_pos = AdaPosScaleNormLayer(dg, act_layer, norm_type='node')

        self.lin_v = nn.Sequential(
            act_layer(),
            nn.Linear(dv, 2 * dv),
        )

        self.lin_v_pos = nn.Linear(dv, de)
        self.lin_e = nn.Sequential(
            act_layer(),
            nn.Linear(de, 2 * de),
        )

        nn.init.zeros_(self.lin_v[-1].weight)
        nn.init.zeros_(self.lin_v[-1].bias)
        nn.init.zeros_(self.lin_e[-1].weight)
        nn.init.zeros_(self.lin_e[-1].bias)

    def forward(self, g, fv, fe, fg, fv_pos, fv_c, fe_c, fv_pos_c):
        shift_v, scale_v = self.lin_v(fv_c + fv_pos_c).chunk(2, dim=1)
        fv = self.norm_v(fv) * (1 + scale_v) + shift_v

        fv_pos_c_ = self.lin_v_pos(fv_pos_c)
        fv2e_pos_c = dgl.ops.gsddmm(g, 'add', fv_pos_c_, fv_pos_c_)
        shift_e, scale_e = self.lin_e(fe_c + fv2e_pos_c).chunk(2, dim=1)
        fe = self.norm_e(fe) * (1 + scale_e) + shift_e

        fv_pos = self.norm_v_pos(g, fv_pos, fg)

        fg = self.norm_g(fg)

        return fv, fe, fg, fv_pos


class ReadoutV(nn.Module):
    def __init__(self, dv, dg):
        """ Map node features to global features """
        super().__init__()
        self.fc1 = nn.Linear(dv, dg)
        self.fc2 = nn.Linear(dv, dg)
        self.fc3 = nn.Linear(dv, dg)

    def forward(self, g: dgl.DGLGraph, fv):
        fv1 = dgl.ops.segment_reduce(g.batch_num_nodes(), fv, 'mean')
        fv2 = dgl.ops.segment_reduce(g.batch_num_nodes(), fv, 'min')
        fv3 = dgl.ops.segment_reduce(g.batch_num_nodes(), fv, 'max')

        out = self.fc1(fv1) + self.fc2(fv2) + self.fc3(fv3)
        return out


class ReadoutE(nn.Module):
    def __init__(self, de, dg):
        """ Map node features to global features """
        super().__init__()
        self.fc1 = nn.Linear(de, dg)
        self.fc2 = nn.Linear(de, dg)
        self.fc3 = nn.Linear(de, dg)

    def forward(self, g: dgl.DGLGraph, fe):
        fe1 = dgl.ops.segment_reduce(g.batch_num_edges(), fe, 'mean')
        fe2 = dgl.ops.segment_reduce(g.batch_num_edges(), fe, 'min')
        fe3 = dgl.ops.segment_reduce(g.batch_num_edges(), fe, 'max')

        out = self.fc1(fe1) + self.fc2(fe2) + self.fc3(fe3)
        return out


class MAttentionLayer(nn.Module):

    def __init__(self, dv, de, dg, de_geo=16, n_heads=4, act_layer=nn.GELU):
        super().__init__()
        self.dv = dv
        self.de = de
        self.dg = dg
        self.de_geo = de_geo
        self.n_heads = n_heads
        self.d_head = dv // n_heads

        # Distance encoding
        self.e_mlp_geo = nn.Sequential(
            nn.Linear(3, de_geo),
            act_layer(),
            nn.Linear(de_geo, de_geo),
            act_layer(),
            nn.Linear(de_geo, de_geo, bias=False)
        )

        self.e_geo_a_mlp = nn.Sequential(
            nn.Linear(de_geo, de_geo),
            act_layer(),
            nn.Linear(de_geo, n_heads),
        )

        self.lin_g2v_update = nn.Linear(dg, 2 * dv)

        self.g_a_mlp = nn.Sequential(
            nn.Linear(dg, dg),
            act_layer(),
            nn.Linear(dg, n_heads, bias=False),
        )
        self.lin_v_update = nn.Linear(dv, 2 * n_heads * self.d_head)

        self.v_msg_mlp = nn.Sequential(
            nn.Linear(dv + de, dv),
            act_layer(),
            nn.Linear(dv, n_heads * self.d_head),
            act_layer(),
        )
        self.act_v_msg = act_layer()
        self.v_update_proj = nn.Linear(n_heads * self.d_head, dv)  # or mlp?

        self.v2e_mul1 = nn.Linear(dv, de)
        self.v2e_mul2 = nn.Linear(dv, de)

        self.e_update_mlp_1 = nn.Sequential(
            nn.Linear(de + de_geo + dg, de),
            act_layer(),
            nn.Linear(de, de),
            act_layer(),
            nn.Linear(de, de * 2),
        )
        self.e_update_mlp_2 = nn.Sequential(
            nn.Linear(de, de),
            act_layer(),
            nn.Linear(de, de),
        )

        assert dg % 2 == 0
        self.v_readout = ReadoutV(dv, dg)
        self.e_readout = ReadoutE(de, dg // 2)
        self.e_geo_readout = ReadoutE(de_geo, dg // 2)
        self.g_update_mlp = nn.Sequential(
            nn.Linear(2 * dg + 1, dg),
            act_layer(),
            nn.Linear(dg, dg),
        )
        self.g_update_gru = nn.GRUCell(dg, dg)

        # Process_pos
        self.mlp_pos_msg = nn.Sequential(
            nn.Linear(de, de),
            act_layer(),
            nn.Linear(de, de),
            act_layer(),
            nn.Linear(de, 1, bias=False)
        )

        # init parameters
        for tensor in [
            self.mlp_pos_msg[-1].weight,
            self.lin_g2v_update.weight,
            self.lin_g2v_update.bias,
            self.lin_v_update.weight,
            self.lin_v_update.bias,
            self.e_update_mlp_1[-1].weight,
            self.e_update_mlp_1[-1].bias,
        ]:
            nn.init.zeros_(tensor)

    def get_fe_geo(self, g, fv_pos):
        fv_dist = torch.linalg.vector_norm(fv_pos, ord=2, dim=-1, keepdim=True)  # nv, 1
        fv2e_pos_diff = dgl.ops.gsddmm(g, 'sub', fv_pos, fv_pos)
        fe_dist = torch.linalg.vector_norm(fv2e_pos_diff,
                                           ord=2, dim=-1, keepdim=True)  # ne, 1

        fe_geo = self.e_mlp_geo(torch.cat((fe_dist,
                                           dgl.ops.copy_u(g, fv_dist),
                                           dgl.ops.copy_v(g, fv_dist)), dim=1))  # ne, de_geo

        return fe_geo, fv2e_pos_diff, fe_dist, fv_dist

    def g2x_update(self, g, fv, fg):
        fg2v_scale, fg2v_shift = dgl.broadcast_nodes(g, self.lin_g2v_update(fg)).chunk(2, dim=1)
        fv = fv * (1 + fg2v_scale) + fg2v_shift
        return fv

    def update_fv(self, g, fv, fe, fg, fe_geo):
        # msg -> sum -> update
        # msg: fv_src, fe, fe_geo
        fv_v_scale, fv_v_shift = einops.rearrange(self.lin_v_update(fv), 'nv (x nh dh) -> x nv nh dh',
                                                  nh=self.n_heads, dh=self.d_head, x=2)

        fe_u_msg = dgl.ops.copy_v(g, fv_v_scale) * einops.rearrange(
            self.v_msg_mlp(torch.cat((dgl.ops.copy_u(g, fv), fe), dim=1)),
            'ne (nh dh) -> ne nh dh', nh=self.n_heads, dh=self.d_head)

        fe_u_msg = self.act_v_msg(fe_u_msg + dgl.ops.copy_v(g, fv_v_shift))  # 如果要改变这里的act，记得顺便去掉shift

        fe_a_geo_gate = self.e_geo_a_mlp(fe_geo).unsqueeze(2)  # ne, nh, 1
        fg2e_gate = dgl.broadcast_edges(g, self.g_a_mlp(fg)).unsqueeze(2)  # ne, nh, 1

        fe_u_msg = fe_u_msg * (fe_a_geo_gate + fg2e_gate)  # ne, nh, dh
        fv = dgl.ops.copy_e_sum(g, fe_u_msg).flatten(start_dim=1)  # nv, nh*dh
        fv = self.v_update_proj(fv)

        return fv

    def update_fe(self, g, fv, fe, fg, fe_geo):
        fv2e_ab = dgl.ops.gsddmm(g, 'mul', self.v2e_mul1(fv), self.v2e_mul2(fv))

        fe_shift, fe_scale = self.e_update_mlp_1(
            torch.cat((fv2e_ab, fe_geo, dgl.broadcast_edges(g, fg)), dim=1)
        ).chunk(2, dim=1)

        fe = self.e_update_mlp_2(fe) * (fe_scale + 1) + fe_shift
        return fe

    def update_fg(self, g, fv, fe, fg, fe_geo):
        fe2g = self.e_readout(g, fe)
        fv2g = self.v_readout(g, fv)
        fe_geo2g = self.e_geo_readout(g, fe_geo)
        num_nodes = g.batch_num_nodes().unsqueeze(1)  # ng, 1
        fg = self.g_update_gru(self.g_update_mlp(torch.cat((fe2g, fv2g, fe_geo2g, num_nodes), dim=1)), fg)
        return fg

    def update_fv_pos(self, g, fe, fv_pos, fv2e_pos_diff, fe_dist, epsilon=1):
        fe_msg = self.mlp_pos_msg(fe) * (fv2e_pos_diff / (fe_dist + epsilon))  # ne, 3
        fv_pos_delta = dgl.ops.copy_e_sum(g, fe_msg)
        fv_pos_delta = x2g2x_sub_mean(g, fv_pos_delta, on_node=True)  # move to zero-centered for stability

        fv_pos = fv_pos + fv_pos_delta
        return fv_pos

    def forward(self, g: dgl.DGLGraph, fv, fe, fg, fv_pos):
        """
        Args:
            g:
            fv:
            fe:
            fg:
            fv_pos:

        Returns: fv_out, fe_out, fg_out, fv_pos_new

        """

        fe_geo, fv2e_pos_diff, fe_dist, fv_dist = self.get_fe_geo(g, fv_pos)

        fv = self.g2x_update(g, fv, fg)
        fv = self.update_fv(g, fv, fe, fg, fe_geo)

        fe = self.update_fe(g, fv, fe, fg, fe_geo)

        fg = self.update_fg(g, fv, fe, fg, fe_geo)
        fv_pos = self.update_fv_pos(g, fe, fv_pos, fv2e_pos_diff, fe_dist)

        return fv, fe, fg, fv_pos


class PocketEncoder(nn.Module):
    def __init__(self,
                 din_v_p, din_e_p,
                 dh_v, dh_e, activation_layer=nn.GELU):
        super().__init__()

        self.mlp_v_p = nn.Sequential(nn.Linear(din_v_p, dh_v), activation_layer(),
                                     nn.Linear(dh_v, dh_v))
        self.mlp_e_p = nn.Sequential(nn.Linear(din_e_p, dh_e), activation_layer(),
                                     nn.Linear(dh_e, dh_e))

    def forward(self, g, fv_p, fe_p, fv_pos_p):
        """
        :param g:
        :param fv:
        :param fe:
        :param fg: (g, din_g)
        :param fv_pos:
        :param fvm_t, fvm_pos_t, fem_t:
        :return:
        """

        fv_p = self.mlp_v_p(fv_p)
        fe_p = self.mlp_e_p(fe_p)

        return fv_p, fe_p, fv_pos_p


class LigandEncoder(nn.Module):
    def __init__(self,
                 din_v, din_e, din_g, din_pos,
                 dh_v, dh_e, dh_g,
                 activation_layer=nn.GELU):
        super().__init__()

        self.mlp_v = nn.Sequential(nn.Linear(din_v, dh_v), activation_layer(),
                                   nn.Linear(dh_v, dh_v))
        self.mlp_e = nn.Sequential(nn.Linear(din_e, dh_e), activation_layer(),
                                   nn.Linear(dh_e, dh_e))
        self.mlp_g = nn.Sequential(nn.Linear(din_g, dh_g), activation_layer(),
                                   nn.Linear(dh_g, dh_g))
        self.mlp_v_pos = nn.Identity()

    def forward(self, g, fv, fe, fg, fv_pos):
        fv = self.mlp_v(fv)
        fe = self.mlp_e(fe)
        fg = self.mlp_g(fg)
        fv_pos = self.mlp_v_pos(fv_pos)
        fv_pos = x2g2x_sub_mean(g, fv_pos, on_node=True)  # for stability

        return fv, fe, fg, fv_pos


class LigandDecoder(nn.Module):
    def __init__(self,
                 din_v, din_e, din_g, din_pos,
                 dh_v, dh_e, dh_g,
                 activation_layer=nn.GELU):
        super().__init__()

        self.mlp_v = nn.Sequential(nn.Linear(dh_v, dh_v), activation_layer(),
                                   nn.Linear(dh_v, din_v))
        self.mlp_e = nn.Sequential(nn.Linear(dh_e, dh_e), activation_layer(),
                                   nn.Linear(dh_e, din_e))
        self.mlp_g = nn.Sequential(nn.Linear(dh_g, dh_g), activation_layer(),
                                   nn.Linear(dh_g, din_g))

    def forward(self, g, fv, fe, fg, fv_pos):
        fv = self.mlp_v(fv)
        fe = self.mlp_e(fe)
        fg = self.mlp_g(fg)

        return fv, fe, fg, fv_pos


class MGraphTransformer(nn.Module):

    def __init__(self, n_layers: int,
                 din_v, din_e, din_g, din_pos,
                 dh_v, dh_e, dh_g, de_geo=32,
                 n_heads=4, norm=True, activation_layer=nn.GELU):
        super().__init__()

        self.noise2v = nn.Sequential(nn.Linear(2, 32), activation_layer(),
                                     nn.Linear(32, dh_v))  # fv有两种特征 todo 进行配置而不是硬编码？
        self.noise2v_pos = nn.Sequential(nn.Linear(1, 32), activation_layer(),
                                         nn.Linear(32, dh_v))
        self.noise2e = nn.Sequential(nn.Linear(1, 32), activation_layer(),
                                     nn.Linear(32, dh_e))

        self.ligand_encoder = LigandEncoder(din_v, din_e, din_g, din_pos, dh_v, dh_e, dh_g, activation_layer)

        layers = []
        for i in range(n_layers):
            layers.append(
                nn.ModuleList([MAttentionLayer(dv=dh_v, de=dh_e, dg=dh_g,
                                               de_geo=de_geo, n_heads=n_heads, act_layer=activation_layer),
                               AdaLN(dh_v, dh_e, dh_g, activation_layer) if norm else nn.Identity(),
                               nn.Linear(dh_v, dh_v),
                               nn.Linear(dh_e, dh_e),
                               nn.Linear(dh_v, 1)])
            )
        self.layers = nn.ModuleList(layers)

        for attn_layer, norm_layer, *gates in layers:
            for i in gates:
                nn.init.zeros_(i.weight)
                nn.init.zeros_(i.bias)

        self.ligand_decoder = LigandDecoder(din_v, din_e, din_g, din_pos, dh_v, dh_e, dh_g, activation_layer)

    def forward_main(self, g, fv, fe, fg, fv_pos, fvm_t, fvm_pos_t, fem_t):
        fvm_t = self.noise2v(fvm_t)
        fvm_pos_t = self.noise2v_pos(fvm_pos_t)
        fem_t = self.noise2e(fem_t)

        fv_c = fvm_t
        fe_c = fem_t
        fv_pos_c = fvm_pos_t
        # main
        for attn_layer, norm_layer, v_gate_layer, e_gate_layer, v_pos_gate_layer in self.layers:
            fv_, fe_, fg_, fv_pos_ = norm_layer(g, fv, fe, fg, fv_pos, fv_c=fv_c, fe_c=fe_c, fv_pos_c=fv_pos_c)
            fv_, fe_, fg_, fv_pos_ = attn_layer(g, fv_, fe_, fg_, fv_pos_)
            fv = fv + fv_ * v_gate_layer(fv_c)
            fe = fe + fe_ * e_gate_layer(fe_c)
            # fg is updated inside attn_layer
            fv_pos = fv_pos + fv_pos_ * v_pos_gate_layer(fv_pos_c)

        return fv, fe, fg, fv_pos

    def forward(self,
                g_l,
                fv_l, fe_l, fg_l, fv_pos_l,
                fvm_t_l, fvm_pos_t_l, fem_t_l,
                ):
        """
        :param g:
        :param fv:
        :param fe:
        :param fg: (g, din_g)
        :param fv_pos:
        :param fvm_t, fvm_pos_t, fem_t:
        :return:
        """

        fv_l, fe_l, fg_l, fv_pos_l = self.ligand_encoder(g_l, fv_l, fe_l, fg_l, fv_pos_l)

        g = g_l
        fv = fv_l
        fv_pos = fv_pos_l
        fe = fe_l
        fg = fg_l

        fvm_t = fvm_t_l
        fvm_pos_t = fvm_pos_t_l
        fem_t = fem_t_l

        # noise level encoding
        fv, fe, fg, fv_pos = self.forward_main(g, fv, fe, fg, fv_pos, fvm_t, fvm_pos_t, fem_t)

        # extract lig features
        fv_l = fv
        fv_pos_l = fv_pos
        fe_l = fe
        fg_l = fg

        fv_l, fe_l, fg_l, fv_pos_l = self.ligand_decoder(g_l, fv_l, fe_l, fg_l, fv_pos_l)

        return fv_l, fe_l, fg_l, fv_pos_l


class PocketGraphTransformer(nn.Module):

    def __init__(self, n_layers: int,
                 din_v, din_e, din_g, din_pos,
                 din_v_p, din_e_p,
                 dh_v, dh_e, dh_g, de_geo=32,
                 n_heads=4, dropout=0, norm=False, activation_layer=nn.GELU):
        super().__init__()

        self.mgt = MGraphTransformer(n_layers, din_v, din_e, din_g, din_pos,
                                     dh_v, dh_e, dh_g, de_geo, n_heads, norm, activation_layer)

        self.pocket_encoder = PocketEncoder(din_v_p, din_e_p, dh_v, dh_e, activation_layer)

    def forward(self,
                g_l,
                fv_l, fe_l, fg_l, fv_pos_l,
                fvm_t_l, fvm_pos_t_l, fem_t_l,
                g_p,
                fv_p, fe_p, fv_pos_p,
                fvm_t_p, fvm_pos_t_p, fem_t_p,
                g_full,
                v_idx, e_idx):
        """
        :param g:
        :param fv:
        :param fe:
        :param fg: (g, din_g)
        :param fv_pos:
        :param fvm_t, fvm_pos_t, fem_t:
        :return:
        """

        fv_l, fe_l, fg_l, fv_pos_l = self.mgt.ligand_encoder(g_l, fv_l, fe_l, fg_l, fv_pos_l)
        fv_p, fe_p, fv_pos_p = self.pocket_encoder(g_p, fv_p, fe_p, fv_pos_p)

        # combine graphs
        g = g_full

        # dataset.__getitem__ determines the order of features required by v_idx/e_idx
        fv = torch.cat([fv_l, fv_p], dim=0)[v_idx]  # lig, pkt
        fv_pos = torch.cat([fv_pos_l, fv_pos_p], dim=0)[v_idx]
        fe_lp = fe_l.new_zeros((g.edata['_TYPE'] == 1).sum(), fe_l.size(1))
        fe = torch.cat([fe_l, fe_lp, fe_p], dim=0)[e_idx]  # lig-lig, lig-pkt, pkt-pkt
        fg = fg_l

        fvm_t = torch.cat([fvm_t_l, fvm_t_p], dim=0)[v_idx]
        fvm_pos_t = torch.cat([fvm_pos_t_l, fvm_pos_t_p], dim=0)[v_idx]
        fem_t_lp = fem_t_l.new_zeros((g.edata['_TYPE'] == 1).sum(), fem_t_l.size(1))
        fem_t = torch.cat([fem_t_l, fem_t_lp, fem_t_p], dim=0)[e_idx]  # todo .contiguous() ?

        fv, fe, fg, fv_pos = self.mgt.forward_main(g, fv, fe, fg, fv_pos, fvm_t, fvm_pos_t, fem_t)

        # extract lig features
        fv_l = fv[g.ndata['_TYPE'] == 0]
        fv_pos_l = fv_pos[g.ndata['_TYPE'] == 0]
        fe_l = fe[g.edata['_TYPE'] == 0]
        fg_l = fg

        fv_l, fe_l, fg_l, fv_pos_l = self.mgt.ligand_decoder(g_l, fv_l, fe_l, fg_l, fv_pos_l)

        return fv_l, fe_l, fg_l, fv_pos_l
