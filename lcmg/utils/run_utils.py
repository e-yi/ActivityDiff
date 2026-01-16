from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from lcmg.models.lcmg_module import _FeatureProcessor
from lcmg.evaluations.sampling import get_sampling_metrics


def no_nan(x):
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()


def sample_from_probs(probs, max=False, n_repeats=1):
    """
    probs to onehot
    """
    n_cls = probs.size(-1)

    if max:
        samples = probs.argmax(1)
        samples = F.one_hot(samples, n_cls).float()  # nx, dx
        return samples

    samples = torch.multinomial(probs, n_repeats, replacement=True)
    counts = probs.new_zeros(probs.size(0), n_cls)
    counts.scatter_add_(1, samples, probs.new_ones(samples.shape))
    samples = counts / n_repeats

    return samples


def new_shift(g, f_0_pred, cls_models, target_labels, _fp, lbd=1, epsilon=1e-12,
              fvm=None, fem=None, input_logit=True, skip_charge=True,
              prot=False, batch_prot_data_list=None, ):
    # 不用log
    fv_grad_accum = None  # torch.zeros_like(fv)
    fe_grad_accum = None  # torch.zeros_like(fe)

    for i, (cls_model, target_label) in enumerate(zip(cls_models, target_labels, strict=True)):
        cls_model.eval()
        cls_model.requires_grad_(False)

        log_f_cat = _fp.concat_features({k: (v - torch.logsumexp(v, dim=1, keepdim=True) if input_logit
                                             else (v + epsilon).log()) for k, v in f_0_pred.items()}, feature_keys=['v', 'e'])
        log_fv0 = log_f_cat['v']
        log_fe0 = log_f_cat['e']

        f_cat = _fp.concat_features({k: (v.softmax(1) if input_logit else v) for k, v in f_0_pred.items()}, feature_keys=['v', 'e'])
        fv0 = f_cat['v']
        fe0 = f_cat['e']
        with torch.set_grad_enabled(True):
            fv = fv0.detach().requires_grad_(True)
            fe = fe0.detach().requires_grad_(True)
            if fvm is not None:
                fv_ = torch.cat([fv, fvm], dim=-1)
                fe_ = torch.cat([fe, fem], dim=-1)
            else:
                fv_ = fv
                fe_ = fe

            if prot:
                batch_prot_data = batch_prot_data_list[i]
                if target_label == 1:
                    outputs = (epsilon + cls_model(g, fv_, fe_, batch_prot_data).sigmoid()).log()
                elif target_label == 0:
                    outputs = (epsilon + (1 - cls_model(g, fv_, fe_, batch_prot_data).sigmoid())).log()
                else:
                    raise NotImplementedError()
            else:
                if target_label == 1:
                    outputs = (epsilon + cls_model(g, fv_, fe_).sigmoid()).log()
                elif target_label == 0:
                    outputs = (epsilon + (1 - cls_model(g, fv_, fe_).sigmoid())).log()
                else:
                    raise NotImplementedError()

            fv_grad, fe_grad = torch.autograd.grad(outputs.sum(), [fv, fe])
            no_nan(fv_grad)
            no_nan(fe_grad)
            if fv_grad_accum is None:
                fv_grad_accum = torch.zeros_like(fv_grad)
            if fe_grad_accum is None:
                fe_grad_accum = torch.zeros_like(fe_grad)
            fv_grad_accum += fv_grad
            fe_grad_accum += fe_grad

    ffv = log_fv0 + fv_grad_accum * lbd
    ffe = log_fe0 + fe_grad_accum * lbd

    no_nan(ffv)
    no_nan(ffe)

    x = _fp.split_features({'v': ffv, 'e': ffe})

    x = {k: v - torch.logsumexp(v, 1, keepdim=True) for k, v in x.items()}  # 归一化

    res = {k: (v if input_logit else (v + epsilon).exp()) for k, v in x.items()}

    if skip_charge:
        res['v_atom_charge'] = f_0_pred['v_atom_charge']
        if input_logit:
            res['v_atom_charge'] = res['v_atom_charge'] - torch.logsumexp(res['v_atom_charge'], 1, keepdim=True)

    return res



def run(self, lbd_fn, cls_models, target_labels, save_n_mid_steps=0, fragment=None, batch_size=64):
    _fp = _FeatureProcessor(self.hparams['train_dataset_info'])

    cls_models = [i.eval() for i in cls_models]

    lig_condition = {}
    g_lig, f_T, _lig_condition = self.sample_noisy_graph(batch_size=batch_size, fragment=fragment)
    g_lig = g_lig.to(self.device)
    f_T = {k: v.to(self.device) for k, v in f_T.items()}
    _lig_condition = {k: v.to(self.device) for k, v in _lig_condition.items()}
    lig_condition.update(_lig_condition)

    save_n_mid_steps = min(self.hparams['num_diffusion_timesteps'] - 1, max(0, save_n_mid_steps))

    if lig_condition is not None:
        f_0 = {i[1:]: v for i, v in lig_condition.items() if i.startswith('f')}
        masks = {i[1:]: v for i, v in lig_condition.items() if i.startswith('m')}
    else:
        f_0 = None
        masks = None

    # t=T
    f_t = f_T

    steps_to_record = np.linspace(self.hparams['num_diffusion_timesteps'], 0, save_n_mid_steps + 2)
    steps_to_record += 0.01
    pointer = 0

    res = {}

    pointer += 1
    res[self.hparams['num_diffusion_timesteps']] = {'v_pos': f_t['v_pos'].cpu(),
                                                    'v_atom_type': f_t['v_atom_type'].argmax(1).cpu(),
                                                    'v_atom_charge': f_t['v_atom_charge'].argmax(1).cpu(),
                                                    'e_bond_type': f_t['e_bond_type'].argmax(1).cpu(), }

    for t_int in tqdm(range(self.hparams['num_diffusion_timesteps'], 0, -1)):
        t = torch.ones((g_lig.batch_size, 1), device=self.device, dtype=torch.long) * t_int

        f_pred = self._step(g_lig, t, f_t, masks=masks, f_0=f_0, maskfill_output=True)

        snrs = {i: self.diffusion_units[i].get_snr(t) for i in
                ['v_pos', 'v_atom_type', 'v_atom_charge', 'e_bond_type']}
        normed_snrs = {k: v / (1 + v) for (k, v) in snrs.items()}  # which turns out to be the `alpha_cumprod`
        signal_rates = self.feature_processor.concat_features_g2x(g_lig, normed_snrs)
        fvm_t = signal_rates['v']
        fem_t = signal_rates['e']

        ## Position 1
        f_pred.update(new_shift(g_lig, f_pred, cls_models=cls_models, target_labels=target_labels, lbd=lbd_fn(t_int),
                                fvm=fvm_t, fem=fem_t, input_logit=True, _fp=_fp))

        # get x_{t-1}
        f_tm1_ = {
            'v_pos': self.diffusion_units['v_pos'].p_sample(model_output=f_pred['v_pos'], x_t=f_t['v_pos'], t=t,
                                                            ef_slice=slice(None),
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
            # return the mean of gaussian distribution
            'v_pos': f_tm1_['v_pos']['mean'],
            # return classes with max probs
            'v_atom_type': f_tm1_['v_atom_type']['probs'],
            'v_atom_charge': f_tm1_['v_atom_charge']['probs'],
            'e_bond_type': f_tm1_['e_bond_type']['probs'],
        }
        if t_int != 1:
            ## Position 2
            ff.update(new_shift(g_lig, ff, cls_models=cls_models, target_labels=target_labels, lbd=lbd_fn(t_int),
                                fvm=fvm_t, fem=fem_t, input_logit=False, _fp=_fp))

        # sample new f_t
        f_t = {
            # return the mean of gaussian distribution
            'v_pos': f_tm1_['v_pos']['samples'],
            # return classes with max probs
            'v_atom_type': sample_from_probs(ff['v_atom_type'], max=True),
            'v_atom_charge': sample_from_probs(ff['v_atom_charge'], max=True),
            'e_bond_type': sample_from_probs(ff['e_bond_type'], max=True),
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

    res_mol = {}
    res_mol[0] = self.batch_f2mols(g_lig, **res[0])

    sampling_metrics = get_sampling_metrics(res_mol[0], self.hparams['train_dataset_info'],
                                            train_smiles=None, metrics=['connectivity',
                                                                        'validity',
                                                                        'uniqueness'],
                                            kekulize=self.hparams['kekulize_sampling_test_mols'])

    f_cat = _fp.concat_features(
        {k: v.cuda() if k == 'v_pos' else sample_from_probs(v.cuda(), max=True) for k, v in ff.items()}
    )

    fv = f_cat['v']
    fe = f_cat['e']

    fv = torch.cat([fv, fvm_t], dim=-1)
    fe = torch.cat([fe, fem_t], dim=-1)

    sampling_metrics.update(
        {
            f'score_{i}': torch.sigmoid(cls_model(g_lig,fv,fe)).cpu().mean().item()
            for i,cls_model in enumerate(cls_models)
        }
    )

    print({k: v for k, v in sampling_metrics.items() if k != 'smiles'})

    sampling_metrics['res_mols'] = res_mol[0]

    return sampling_metrics