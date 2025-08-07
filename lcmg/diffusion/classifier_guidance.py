"""
Classifier Guidance for Diffusion Models

This module contains classifier guidance functionality including the new_shift function
from main.ipynb and guidance schedule functions.
"""

import numpy as np
import torch
from tqdm.auto import tqdm

from lcmg.runtime_utils import pylogger
from lcmg.evaluations.sampling import get_sampling_metrics

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def no_nan(x):
    """Check for NaN values in tensor"""
    assert not torch.isnan(x).any(), "Found NaN values!"
    assert not torch.isinf(x).any(), "Found Inf values!"


def sample_from_probs(probs, max=False, n_repeats=1):
    """Sample from probability distributions"""
    n_cls = probs.size(-1)

    if max:
        samples = probs.argmax(1)
        samples = torch.nn.functional.one_hot(samples, n_cls).float()
        return samples

    samples = torch.multinomial(probs, n_repeats, replacement=True)
    counts = probs.new_zeros(probs.size(0), n_cls)
    counts.scatter_add_(1, samples, probs.new_ones(samples.shape))
    samples = counts / n_repeats
    return samples


# Guidance schedule functions
def cos_fn(i, a=1, b=100, t=505):
    """Cosine guidance schedule - starts high, ends low"""
    s = (np.cos(np.linspace(0, np.pi, t)) + 1) / 2 * (b - a) + a
    return s[t - i]


def lin_fn(i, a=1, b=100, t=505):
    """Linear guidance schedule - starts high, ends low"""
    s = np.linspace(b, a, t)
    return s[t - i]


def val_fn(i, a=100, t=505):
    """Constant guidance schedule"""
    return a


def new_shift(g, f_0_pred, cls_models, target_labels, lbd=1, epsilon=1e-12,
              fvm=None, fem=None, input_logit=True, skip_charge=True,
              feature_processor=None):
    """
    Apply classifier guidance to shift predicted features towards desired properties.

    Args:
        g: Graph
        f_0_pred: Predicted features
        cls_models: List of classifier models
        target_labels: List of target labels (0 or 1)
        lbd: Guidance strength
        epsilon: Small value to avoid log(0)
        fvm: Vertex feature mask
        fem: Edge feature mask
        input_logit: Whether input features are logits
        skip_charge: Whether to skip charge guidance
        feature_processor: Feature processor instance

    Returns:
        Dictionary of guided features
    """
    fv_grad_accum = None
    fe_grad_accum = None

    for i, (cls_model, target_label) in enumerate(zip(cls_models, target_labels, strict=True)):
        cls_model.eval()
        cls_model.requires_grad_(False)

        log_f_cat = feature_processor.concat_features({k: (v-torch.logsumexp(v, dim=1, keepdim=True) if input_logit
                                                           else (v+epsilon).log()) for k, v in f_0_pred.items()},
                                                      feature_keys=['v', 'e'])
        log_fv0 = log_f_cat['v']
        log_fe0 = log_f_cat['e']

        f_cat = feature_processor.concat_features({k: (v.softmax(1) if input_logit else v) for k, v in f_0_pred.items()},
                                                  feature_keys=['v', 'e'])
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

            if target_label == 1:
                outputs = (epsilon + cls_model(g, fv_, fe_).sigmoid()).log()
            elif target_label == 0:
                outputs = (
                    epsilon + (1-cls_model(g, fv_, fe_).sigmoid())).log()
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

    ffv = log_fv0 + fv_grad_accum*lbd
    ffe = log_fe0 + fe_grad_accum*lbd

    no_nan(ffv)
    no_nan(ffe)

    x = feature_processor.split_features({'v': ffv, 'e': ffe})

    x = {k: v-torch.logsumexp(v, 1, keepdim=True) for k, v in x.items()}  # 归一化

    res = {k: (v if input_logit else (v+epsilon).exp()) for k, v in x.items()}

    if skip_charge:
        res['v_atom_charge'] = f_0_pred['v_atom_charge']
        if input_logit:
            res['v_atom_charge'] = res['v_atom_charge'] - \
                torch.logsumexp(res['v_atom_charge'], 1, keepdim=True)

    return res
