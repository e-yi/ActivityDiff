from typing import Any, Optional

import dgl
import numpy as np
import torch
from lightning.pytorch import LightningModule
from packmetric import MetricGroup
from rdkit import Chem
from tqdm.auto import tqdm


from lcmg.data.datasets.utils.mol2graph import build_complete_graphs, build_graphs_with_fragment
from lcmg.evaluations.sampling import get_sampling_metrics
from lcmg.runtime_utils import pylogger
from lcmg.utils.g_utils import apply_r_t, get_align_r_t
from lcmg.utils.mol_utils import build_mol
from lcmg.diffusion.classifier_guidance import new_shift, sample_from_probs

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class _FeatureProcessor:

    def __init__(self, info):
        f_structure = {
            'v': {'v_atom_type': slice(0, len(info['at2idx'])),
                  'v_atom_charge': slice(len(info['at2idx']), len(info['at2idx'])+len(info['ac2idx']))},
            'v_pos': {'v_pos': slice(0, info['n_pos_dim'])},
            'e': {'e_bond_type': slice(0, len(info['bt2idx']))},
        }  # used to reconstruct different type of features from node/edge features

        self.f_structure = f_structure
        self.keys = f_structure.keys()

    def concat_features(self, f_t, f_0=None, masks=None, feature_keys='all'):
        """
        Concatenate features according to the feature structure.

        :param f_t: Current feature tensor
        :param f_0: Initial feature tensor for masking (optional)
        :param masks: Dictionary of masks for features (optional)
        :param feature_keys: Iterable specifying which features to concatenate ('v', 'e', 'v_pos'). Default is 'all'.
        :return: Dictionary of concatenated features.
        """
        f_structure = self.f_structure

        f_t_cat = {}

        if feature_keys == 'all':
            feature_keys = self.keys

        for fsname in feature_keys:
            cur_feats = []
            for f_name in f_structure[fsname]:
                f = f_t[f_name]

                if f_0 is not None and masks is not None and f_name in masks:
                    mask = masks[f_name].unsqueeze(1)

                    f_mask = torch.ones_like(f) * mask
                    f[f_mask == 1] = f_0[f_name][f_mask == 1]

                cur_feats.append(f)

            f_t_cat[fsname] = torch.cat(cur_feats, dim=-1)

        return f_t_cat

    def concat_features_g2x(self, g, fg_, masks=None, feature_keys='all'):
        """
        Concatenate features for graph using dgl.broadcast.

        :param g: Graph object
        :param fg_: Feature dictionary
        :param masks: Dictionary of masks for features (optional)
        :param feature_keys: Iterable specifying which features to concatenate ('v', 'e', 'v_pos'). Default is 'all'.
        :return: Dictionary of signal rates.
        """
        f_structure = self.f_structure

        signal_rates = {}

        if feature_keys == 'all':
            feature_keys = self.keys

        for fsname in feature_keys:
            cur_signal_rate = []
            for f_name in f_structure[fsname]:

                bfn = dgl.broadcast_edges if fsname == 'e' else dgl.broadcast_nodes
                signal_rate = bfn(g, fg_[f_name])

                if masks is not None and f_name in masks:
                    mask = masks[f_name].unsqueeze(1)
                    n_mask = torch.ones_like(signal_rate) * mask
                    signal_rate[n_mask == 1] = 1

                cur_signal_rate.append(signal_rate)

            signal_rates[fsname] = torch.cat(cur_signal_rate, dim=-1)

        return signal_rates

    def split_features(self, f):
        """
        Split concatenated features back into original components.

        :param f: Concatenated feature tensor
        :return: Dictionary of split features.
        """
        f_structure = self.f_structure

        split_f = {
            f_name: f[fsname][..., slice_j] for fsname in f for f_name, slice_j in f_structure[fsname].items()
        }

        return split_f


class LCMGModule(LightningModule):
    """
    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    """
    diffusion over
        - atom types  Categorical
        - atom charges  Categorical
        - bond types  Categorical
    """

    def __init__(
            self,
            net,  # partial
            train_dataset_info,
            num_diffusion_timesteps,
            diffusion_units: dict,  # partial
            lambdas,  # weights of losses
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            do_sample_freq,
            compile,
            learning_rate,
            warmup_steps,
            fg_init_type='normed_snr',  # Only support normed_snr for open source
            fg_in_dim=4,
            snr_gamma=5,  # Fixed value from config
            metric_group: Optional[MetricGroup] = None,
            kekulize_sampling_test_mols=True,  # Fixed value from config
    ):
        super().__init__()

        # Fixed configuration for open source version
        assert fg_init_type == 'normed_snr', "Only 'normed_snr' is supported in open source version"
        din_g = 4  # Fixed for normed_snr

        assert snr_gamma > 0, "SNR weighting is required in open source version"

        self.save_hyperparameters(logger=False, ignore=['metric_group'])

        net = net(din_v=len(train_dataset_info['atom_types']) + len(train_dataset_info['atom_charges']),
                  din_e=len(train_dataset_info['bond_types']),
                  din_g=din_g,
                  din_pos=train_dataset_info['n_pos_dim'])
        self.net = net

        diffusion_units_init = {}  # create a new dict to avoid changing self.hparam
        # v_pos, v_atom_type, v_atom_charge, e_bond_type
        for k, v in list(diffusion_units.items()):
            if k == 'v_pos':
                diffusion_units_init[k] = v()
            else:
                diffusion_units_init[k] = v(
                    class_probs=train_dataset_info[f'freq_{k.split("_", 1)[1]}'])
        self.diffusion_units = diffusion_units_init

        # none_weight is disabled in open source version
        self.bond_cls_weight = None

        self.metric_group = metric_group if metric_group is not None else MetricGroup([
        ])

        self.feature_processor = _FeatureProcessor(train_dataset_info)

    def _step(self,
              g_lig,
              t,
              f_t,
              *,
              f_0=None,
              masks=None,
              normed_snrs=None,
              maskfill_output=False,
              return_pos_signal_rate=False,
              ):
        """
        todo  显然可以优化速度，移动一些到dataset部分，区分训练和推理
        todo  更灵活的输入：直接输入 signal rate
        Args:
            f_0, mask: if provided, replace `f_t` with `f_0` where `mask==1`
            maskfill_output: whether to replace `f_pred` with `f_0` where `mask==1`
        """
        if normed_snrs is None:
            snrs = {i: self.diffusion_units[i].get_snr(t) for i in
                    ['v_pos', 'v_atom_type', 'v_atom_charge', 'e_bond_type']}
            # which turns out to be the `alpha_cumprod`
            normed_snrs = {k: v / (1 + v) for (k, v) in snrs.items()}

        # init fg
        match self.hparams['fg_init_type']:
            case 't':
                fg_t = torch.ones((g_lig.batch_size, self.net.din_g), device=g_lig.device) * (
                    t / self.hparams['num_diffusion_timesteps'])
            case 'normed_snr':
                fg_t = torch.cat([i for i in normed_snrs.values()], dim=1)
            case 'ones':
                fg_t = torch.ones(
                    (g_lig.batch_size, self.net.din_g), device=g_lig.device)
            case _:
                raise NotImplementedError(self.hparams['fg_init_type'])

        f_t_cat = self.feature_processor.concat_features(f_t, f_0, masks)
        fv_t = f_t_cat['v']
        fv_pos_t = f_t_cat['v_pos']
        fe_t = f_t_cat['e']

        signal_rates = self.feature_processor.concat_features_g2x(
            g_lig, normed_snrs, masks)
        fvm_t = signal_rates['v']
        fvm_pos_t = signal_rates['v_pos']
        fem_t = signal_rates['e']

        # use_pkt is disabled in open source version
        fv_pred, fe_pred, fg_pred, fv_pos_pred = self.forward(g_lig, fv_t, fe_t, fg_t, fv_pos_t,
                                                              fvm_t, fvm_pos_t, fem_t)

        assert fv_pred.shape == fv_t.shape
        assert fv_pos_pred.shape == fv_pos_t.shape
        assert fe_pred.shape == fe_t.shape

        f_pred = self.feature_processor.split_features(
            {'v': fv_pred, 'v_pos': fv_pos_pred, 'e': fe_pred})

        if maskfill_output:
            if f_0 is not None and masks is not None:
                for name in f_pred:
                    if name in masks:
                        mask = masks[name]
                        f_pred[name][mask] = f_0[name][mask] * 10000

        if return_pos_signal_rate:
            return f_pred, signal_rates['v_pos']
        else:
            return f_pred

    def model_step(self, t, g_lig, lig_features):

        f_0 = {i[1:]: v for i, v in lig_features.items() if i.startswith('f')}
        masks = {i[1:]: v for i, v in lig_features.items()
                 if i.startswith('m')}

        # get q(x_t|x_0)

        f_t = {
            # gaussian diffusion
            'v_pos': self.diffusion_units['v_pos'].q_sample(x_start=f_0['v_pos'],
                                                            t=t, g=g_lig, on_node=True,
                                                            ef_slice=slice(None)),
            # categorical diffusion
            'v_atom_type': self.diffusion_units['v_atom_type'].q_sample(x_start=f_0['v_atom_type'],
                                                                        t=t, g=g_lig, on_node=True),
            'v_atom_charge': self.diffusion_units['v_atom_charge'].q_sample(x_start=f_0['v_atom_charge'],
                                                                            t=t, g=g_lig, on_node=True),
            'e_bond_type': self.diffusion_units['e_bond_type'].q_sample(x_start=f_0['e_bond_type'],
                                                                        t=t, g=g_lig, on_node=False),
        }

        snrs = {i: self.diffusion_units[i].get_snr(
            t) for i in ['v_pos', 'v_atom_type', 'v_atom_charge', 'e_bond_type']}
        # which turns out to be the `alpha_cumprod`
        normed_snrs = {k: v / (1 + v) for (k, v) in snrs.items()}

        # use_pkt is disabled in open source version
        f_pred, pos_signal_rate = self._step(g_lig, t, f_t, masks=masks, f_0=f_0, normed_snrs=normed_snrs,
                                             return_pos_signal_rate=True)

        # use_aligned_loss is enabled in open source version
        weights = pos_signal_rate.squeeze(1)
        _r, _t = get_align_r_t(
            pos=f_0['v_pos'], ref_pos=f_pred['v_pos'], weights=weights, graph=g_lig)
        aligned_pos_0 = apply_r_t(graph=g_lig, pos=f_0['v_pos'], r=_r, t=_t)
        f_0['v_pos'] = aligned_pos_0

        # get losses
        res_dict = {}

        # todo 这里的mask_sigma可以改得更加细粒度，不考虑mask而直接考虑某个节点上的某个特征的置信度（snr）
        losses = {
            i: self.diffusion_units[i].training_losses(model_output=f_pred[i], x_start=f_0[i], x_t=f_t[i],
                                                       x_mask=masks[i],
                                                       t=t, g=g_lig, on_node=i.startswith(
                                                           'v'),
                                                       mask_sigma=self.hparams['mask_sigma'])['loss']
            for i in ['v_pos', 'v_atom_type', 'v_atom_charge']
        }

        i = 'e_bond_type'
        losses[i] = self.diffusion_units[i].training_losses(
            model_output=f_pred[i],
            x_start=f_0[i], x_t=f_t[i],
            x_mask=masks[i],
            t=t, g=g_lig,
            on_node=i.startswith('v'),
            weight=self.bond_cls_weight.to(
                f_t[i].device) if self.bond_cls_weight is not None else None,
            mask_sigma=self.hparams['mask_sigma']
        )['loss']

        snr_gamma = self.hparams['snr_gamma']
        if snr_gamma > 0:
            # use Min-Max-SNR-gamma
            weighted_losses = {
                k: v * torch.clamp(snrs[k], max=snr_gamma) for (k, v) in losses.items()}
        else:
            weighted_losses = losses

        losses = {k: v.mean() for (k, v) in losses.items()}
        weighted_losses = {k: v.mean() for (k, v) in weighted_losses.items()}

        loss = sum([self.hparams['lambdas'][k] * v for (k, v)
                   in weighted_losses.items()], 0)

        assert not torch.isnan(loss)

        res_dict.update({
            'loss': loss,  # will be used to update the gradient, the others will be detached
        })

        res_dict.update({f'loss_{k}': v.item() for (k, v) in losses.items()})
        res_dict.update({f'loss_{k}_weighted': v.item()
                        for (k, v) in weighted_losses.items()})

        return res_dict

    def batch_f2mols(self, g, v_pos, v_atom_type, v_atom_charge, e_bond_type):
        batch_num_nodes = g.batch_num_nodes().tolist()
        batch_num_edges = g.batch_num_edges().tolist()

        pos = v_pos.split(batch_num_nodes)
        atom_type_idx = v_atom_type.split(batch_num_nodes)
        charge_idx = v_atom_charge.split(batch_num_nodes)
        bond_type_idx = e_bond_type.split(batch_num_edges)

        res = []
        for g_i, pos_i, atom_type_idx_i, charge_idx_i, bond_type_idx_i in \
                zip(dgl.unbatch(g), pos, atom_type_idx, charge_idx, bond_type_idx):
            atom_type_i = [self.hparams['train_dataset_info']
                           ['idx2at'][i] for i in atom_type_idx_i.tolist()]
            charge_i = [self.hparams['train_dataset_info']['idx2ac'][i]
                        for i in charge_idx_i.tolist()]
            pos_i = pos_i.numpy()
            bond_type_i = [self.hparams['train_dataset_info']
                           ['idx2bt'][i] for i in bond_type_idx_i.tolist()]

            mol = build_mol(g_i, atom_type_i, charge_i, pos_i, bond_type_i)
            res.append(mol)

        return res

    def sample_noisy_graph(self, n_node_list: Optional = None, batch_size=None, fragment=None):

        if fragment is None:
            min_nodes = 1
        else:
            # for example, `[*]CO[*]` has a minimum of 4 nodes
            min_nodes = fragment.GetNumAtoms()
            if self.hparams['train_dataset_info']['kekulize']:
                fragment = Chem.Mol(fragment)
                Chem.Kekulize(fragment, clearAromaticFlags=True)

        if n_node_list is None:
            if batch_size is None:
                raise ValueError(
                    'invalid parameters: `n_nodes` and `batch_size` are both None')
            n_node_range = list(range(len(self.hparams['train_dataset_info']['freq_nv'])))[
                min_nodes:]
            n_node_probs = np.array(
                self.hparams['train_dataset_info']['freq_nv'][min_nodes:])
            n_node_probs /= n_node_probs.sum()
            n_node_list = np.random.choice(n_node_range,
                                           size=batch_size,
                                           replace=True,
                                           p=n_node_probs)
        else:
            n_node_list = [max(i, min_nodes) for i in n_node_list]

        lig_condition = {}
        if fragment is None:
            graphs = build_complete_graphs(n_node_list).to(self.device)
        else:
            graphs = build_graphs_with_fragment(n_node_list, self.hparams['train_dataset_info'], fragment).to(
                self.device)
            for k in list(graphs.ndata.keys()):
                lig_condition[k] = graphs.ndata.pop(k)
            for k in list(graphs.edata.keys()):
                lig_condition[k] = graphs.edata.pop(k)

        num_nodes = graphs.num_nodes()

        f_T = {
            # gaussian diffusion
            'v_pos': self.diffusion_units['v_pos'].sample_x_T(shape=(num_nodes, 3), device=self.device,
                                                              ef_slice=slice(
                                                                  None),
                                                              g=graphs, on_node=True),
            # categorical diffusion
            'v_atom_type': self.diffusion_units['v_atom_type'].sample_x_T(graphs, on_node=True),
            'v_atom_charge': self.diffusion_units['v_atom_charge'].sample_x_T(graphs, on_node=True),
            'e_bond_type': self.diffusion_units['e_bond_type'].sample_x_T(graphs, on_node=False),
        }

        return graphs, f_T, lig_condition

    @torch.no_grad()
    def batch_sample(self, g_lig, f_T, *, lig_condition=None, save_n_mid_steps=0,
                     cls_models=None, target_labels=None, lbd_fn=None, sample_max=True):
        """
        Args:
            lig_condition: conditions like fragments
            cls_models: List of classifier models for guidance (optional)
            target_labels: List of target labels for guidance (optional)
            lbd_fn: Lambda function for guidance strength (optional)
        """
        save_n_mid_steps = min(
            self.hparams['num_diffusion_timesteps'] - 1, max(0, save_n_mid_steps))

        if lig_condition is not None:
            f_0 = {i[1:]: v for i, v in lig_condition.items()
                   if i.startswith('f')}
            masks = {i[1:]: v for i, v in lig_condition.items()
                     if i.startswith('m')}
        else:
            f_0 = None
            masks = None

        # Check if classifier guidance is enabled
        use_guidance = (
            cls_models is not None and target_labels is not None and lbd_fn is not None)

        # t=T
        f_t = f_T

        steps_to_record = np.linspace(
            self.hparams['num_diffusion_timesteps'], 0, save_n_mid_steps + 2)
        steps_to_record += 0.01
        pointer = 0

        res = {}

        pointer += 1
        res[self.hparams['num_diffusion_timesteps']] = {'v_pos': f_t['v_pos'].cpu(),
                                                        'v_atom_type': f_t['v_atom_type'].argmax(1).cpu(),
                                                        'v_atom_charge': f_t['v_atom_charge'].argmax(1).cpu(),
                                                        'e_bond_type': f_t['e_bond_type'].argmax(1).cpu(), }

        for t_int in tqdm(range(self.hparams['num_diffusion_timesteps'], 0, -1)):
            t = torch.ones((g_lig.batch_size, 1),
                           device=self.device, dtype=torch.long) * t_int

            f_pred = self._step(g_lig, t, f_t, masks=masks, f_0=f_0)

            # Apply classifier guidance to model predictions
            if use_guidance:
                snrs = {i: self.diffusion_units[i].get_snr(t) for i in
                        ['v_pos', 'v_atom_type', 'v_atom_charge', 'e_bond_type']}
                normed_snrs = {k: v / (1 + v) for (k, v) in snrs.items()}
                signal_rates = self.feature_processor.concat_features_g2x(
                    g_lig, normed_snrs)
                fvm_t = signal_rates['v']
                fem_t = signal_rates['e']

                # Apply guidance to f_pred
                f_pred.update(new_shift(g_lig, f_pred, cls_models=cls_models, target_labels=target_labels,
                                        lbd=lbd_fn(t_int), fvm=fvm_t, fem=fem_t, input_logit=True,
                                        feature_processor=self.feature_processor))

            # get x_{t-1}
            f_tm1_ = {
                'v_pos': self.diffusion_units['v_pos'].p_sample(model_output=f_pred['v_pos'], x_t=f_t['v_pos'], t=t,
                                                                ef_slice=slice(
                                                                    None),
                                                                g=g_lig, on_node=True),
                'v_atom_type': self.diffusion_units['v_atom_type'].p_sample(model_output=f_pred['v_atom_type'],
                                                                            x_t=f_t['v_atom_type'], t=t,
                                                                            g=g_lig, on_node=True),
                'v_atom_charge': self.diffusion_units['v_atom_charge'].p_sample(model_output=f_pred['v_atom_charge'],
                                                                                x_t=f_t['v_atom_charge'], t=t,
                                                                                g=g_lig, on_node=True),
                'e_bond_type': self.diffusion_units['e_bond_type'].p_sample(model_output=f_pred['e_bond_type'],
                                                                            x_t=f_t['e_bond_type'], t=t,
                                                                            g=g_lig, on_node=False),
            }

            ff = {
                'v_pos': f_tm1_['v_pos']['mean'],
                'v_atom_type': f_tm1_['v_atom_type']['probs'],
                'v_atom_charge': f_tm1_['v_atom_charge']['probs'],
                'e_bond_type': f_tm1_['e_bond_type']['probs'],
            }

            # Apply classifier guidance to sampled features
            if use_guidance and t_int != 1:
                ff.update(new_shift(g_lig, ff, cls_models=cls_models, target_labels=target_labels,
                                    lbd=lbd_fn(t_int), fvm=fvm_t, fem=fem_t, input_logit=False,
                                    feature_processor=self.feature_processor))

            f_t = {
                'v_pos': f_tm1_['v_pos']['samples'],
                'v_atom_type': sample_from_probs(ff['v_atom_type'], max=sample_max),
                'v_atom_charge': sample_from_probs(ff['v_atom_charge'], max=sample_max),
                'e_bond_type': sample_from_probs(ff['e_bond_type'], max=sample_max),
            }

            if t_int - 1 < steps_to_record[pointer]:
                pointer += 1
                res[t_int - 1] = {
                    # return the mean of gaussian distribution
                    'v_pos': f_t['v_pos'].cpu(),
                    # return classes with max probs
                    'v_atom_type': f_t['v_atom_type'].argmax(1).cpu(),
                    'v_atom_charge': f_t['v_atom_charge'].argmax(1).cpu(),
                    'e_bond_type': f_t['e_bond_type'].argmax(1).cpu(),
                }

        for i in list(res.keys()):
            res[i] = self.batch_f2mols(g_lig, **res[i])

        return res

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def training_step(self, batch: Any, batch_idx: int):
        lig_graphs = batch['lig']['graph']

        t = torch.randint(1, self.hparams['num_diffusion_timesteps'] + 1, size=(lig_graphs.batch_size, 1),
                          device=lig_graphs.device)  # todo test using the same t within a batch

        lig = batch['lig']
        pkt = batch.get('pkt', None)
        full = batch.get('full', None)

        res_dict = self.model_step(
            t,
            g_lig=lig['graph'],
            lig_features=lig['feature'],
            g_pkt=pkt['graph'] if pkt is not None else None,
            pkt_features=pkt['feature'] if pkt is not None else None,
            g_full=full['graph'] if full is not None else None,
            v_idx=full['feature']['nid'] if full is not None else None,
            e_idx=full['feature']['eid'] if full is not None else None,
        )

        return res_dict  # res_dict must contain "loss":loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # todo 找到一个更好的实现方式，支持batch更新的warmup策略与epoch更新的其他scheduler？
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step +
                           1) / self.hparams['warmup_steps'])
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams['learning_rate']

        optimizer.step(closure=optimizer_closure)
        # optimizer.zero_grad()

    def sample_mols(self,
                    *,
                    batch_size=None,
                    save_n_mid_steps=0,
                    lig_condition=None,
                    fragment=None,
                    cls_models=None,
                    target_labels=None,
                    lbd_fn=None,
                    ):
        """

        :param batch_size: number of molecules to generate
        :param save_n_mid_steps: the number of middle states to be saved and returned
        :param lig_condition: conditions like fragments and others
        :param fragment: rdkit.Chem.Mol
        :param cls_models: List of classifier models for guidance (optional)
        :param target_labels: List of target labels (0 or 1) for guidance (optional)
        :param lbd_fn: Lambda function for guidance strength (optional)
        :return: dict[int, rdkit.Chem.RDMol], the key is diffusion_timestep, 0 for the last step results
        """
        self.eval()

        try:
            if not self.trainer.is_global_zero:
                return {}
        except RuntimeError:  # not attached to a trainer
            pass

        if lig_condition is not None:
            assert fragment is None
        else:
            lig_condition = {}

        g_l, f_T, _lig_condition = self.sample_noisy_graph(
            batch_size=batch_size, fragment=fragment)
        g_l = g_l.to(self.device)
        f_T = {k: v.to(self.device) for k, v in f_T.items()}
        _lig_condition = {k: v.to(self.device)
                          for k, v in _lig_condition.items()}
        lig_condition.update(_lig_condition)

        res = self.batch_sample(
            g_l, f_T, save_n_mid_steps=save_n_mid_steps, lig_condition=lig_condition,
            cls_models=cls_models, target_labels=target_labels, lbd_fn=lbd_fn)

        return res

    def validation_step(self, batch: Any, batch_idx: int):
        lig_graphs = batch['lig']['graph']

        t = torch.randint(1, self.hparams['num_diffusion_timesteps'] + 1, size=(lig_graphs.batch_size, 1),
                          device=lig_graphs.device)

        lig = batch['lig']
        pkt = batch.get('pkt', None)
        full = batch.get('full', None)

        res_dict = self.model_step(
            t,
            g_lig=lig['graph'],
            lig_features=lig['feature'],
            g_pkt=pkt['graph'] if pkt is not None else None,
            pkt_features=pkt['feature'] if pkt is not None else None,
            g_full=full['graph'] if full is not None else None,
            v_idx=full['feature']['nid'] if full is not None else None,
            e_idx=full['feature']['eid'] if full is not None else None,
        )

        if self.hparams['do_sample_freq'] > 0 and batch_idx == 0 \
                and self.current_epoch % self.hparams['do_sample_freq'] == 0:
            if pkt is not None:
                f_p = {i[1:]: v for i, v in batch['pkt']
                       ['feature'].items() if i.startswith('f')}
                res = self.sample_mols(
                    g_p=batch['pkt']['graph'],
                    f_p=f_p,
                )
            else:
                res = self.sample_mols(batch_size=128)

            sampling_metrics = get_sampling_metrics(res[0], self.hparams['train_dataset_info'],
                                                    train_smiles=None, metrics='all',
                                                    kekulize=self.hparams['kekulize_sampling_test_mols'])

            log.debug(sampling_metrics)
            res_dict['sampling_metrics'] = sampling_metrics

        return res_dict

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Called in the validation loop after the batch.

        Args:
            outputs: The outputs of validation_step(x)
            batch: The batched data as it is returned by the validation DataLoader.
            batch_idx: the index of the batch
            dataloader_idx: the index of the dataloader

        """

        graphs = batch['lig']['graph']

        if 'sampling_metrics' in outputs and self.logger is not None:
            sampling_metrics = outputs['sampling_metrics']
            # wandb logger
            self.logger.log_text(key='samples', columns=['smiles'], data=[
                                 [i] for i in sampling_metrics['smiles']])
            self.log_dict({f'sampling/{k}': v for (k, v) in sampling_metrics.items() if k != 'smiles'},
                          batch_size=graphs.batch_size)

    def test_step(self, batch: Any, batch_idx: int):
        lig_graphs = batch['lig']['graph']

        t = torch.randint(1, self.hparams['num_diffusion_timesteps'] + 1, size=(lig_graphs.batch_size, 1),
                          device=lig_graphs.device)

        lig = batch['lig']
        pkt = batch.get('pkt', None)
        full = batch.get('full', None)

        res_dict = self.model_step(
            t,
            g_lig=lig['graph'],
            lig_features=lig['feature'],
            g_pkt=pkt['graph'] if pkt is not None else None,
            pkt_features=pkt['feature'] if pkt is not None else None,
            g_full=full['graph'] if full is not None else None,
            v_idx=full['feature']['nid'] if full is not None else None,
            e_idx=full['feature']['eid'] if full is not None else None,
        )

        return res_dict

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_epoch",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
