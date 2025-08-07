# Part of code adapted from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py

import numpy as np
import torch

from lcmg.diffusion.diffusion_base import ModelPredType
from lcmg.utils.g_utils import g2x_copy, x2g2x_sub_mean, x2g_sum


class ENNormalDiffusion:

    def __init__(self, *, betas, model_pred_type: ModelPredType):
        """
        normal diffusion with E(n) equivalent support (through zero-centered noise)
        following midi, currently only support use x_0 as the target. todo support epsilon

        loss_type: MSE only
        variance_type: Fixed only

        Args:
            betas:
        """
        if model_pred_type not in [ModelPredType.START_X, ModelPredType.PREVIOUS_X]:
            raise NotImplementedError()

        self.model_pred_type = model_pred_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        # self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        # self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        # self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

    def sample_x_T(self, shape, device, ef_slice, g, on_node):
        """

        Args:
            shape:
            device:
            ef_slice: indicis of zero-centered noises
            g:
            on_node:

        Returns:
            sample (nx,dx)
        """
        return self.sample_noise(shape, device, ef_slice, g, on_node)

    def sample_noise(self, shape, device, ef_slice, g, on_node):
        """

        Args:
            shape:
            device:
            ef_slice: slice, currently only support a group of features to be E(n) eq, e.g. a group of positions

            g:
            on_node:

        Returns:

        """
        noise = torch.randn(shape, device=device)
        if ef_slice is not None:
            noise[..., ef_slice] = x2g2x_sub_mean(g, noise[..., ef_slice], on_node=on_node)
        return noise

    def get_snr(self, t, min=1e-6, max=1e6):
        if (t<=0).any():
            raise ValueError(f't needs to be greater than 0')

        alpha_cumprod_t = torch.from_numpy(self.alphas_cumprod).to(t.device)[t - 1].float()  # ng, 1
        snr = alpha_cumprod_t / (1 - alpha_cumprod_t)
        return snr.clamp(min=min, max=max)

    def q_sample(self, *, x_start, t, g, on_node, noise=None, ef_slice=slice(None)):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        Args:
            x_start: the initial data batch.
            t: the number of diffusion steps, minus 1 to get the correct array index
            noise: if specified, the split-out normal noise.
            ef_slice: slice, currently only support a group of features to be E(n) eq, e.g. a group of positions

        Returns:
            A noisy version of x_start.
        """
        if (t <= 0).any():
            raise ValueError(f't needs to be greater than 0')

        if noise is None:
            noise = self.sample_noise(x_start.shape, x_start.device, ef_slice, g, on_node)

        assert noise.shape == x_start.shape
        tm1 = t - 1  # minus 1 to get the correct array index
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, tm1, x_start.shape, g, on_node) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, tm1, x_start.shape, g, on_node)
                * noise
        )

    def training_losses(self, *, model_output, x_start, x_t, t, g, on_node: bool, x_mask=None, mask_sigma=0):
        """
        todo: support elision
        Args:
            model_output: shape=(dx, f_dim)
            x_start:  shape=(dx, f_dim)
            x_t: shape=(dx, f_dim)
            t: shape=(dg, 1)  the number of diffusion steps, minus 1 to get the correct array index
            on_node: is model_output node features or edge features
            noise:
            x_mask: loss values where x_mask is 0 will be kept

        Returns:
            a dict that contains following keys:
                - 'loss'
                - 'loss_type'
        """
        assert model_output.dim() == x_start.dim() == 2

        if (t <= 0).any():
            raise ValueError(f't needs to be greater than 0')

        terms = {}

        if self.model_pred_type == ModelPredType.START_X:
            target = x_start
        elif self.model_pred_type == ModelPredType.PREVIOUS_X:
            target, _, _ = self._q_posterior_mean_variance(x_start, x_t, t, g, on_node)
        else:
            raise NotImplementedError(self.model_pred_type)

        assert model_output.shape == target.shape == x_start.shape

        terms['loss_type'] = 'masked_rmse'

        rmse = torch.nn.functional.mse_loss(model_output, target, reduction='none').sum(1).sqrt()

        flat_reversed_mask = (~x_mask.bool()).view(-1).float()
        flat_reversed_mask[flat_reversed_mask == 0] = mask_sigma
        rmse_masked_sum = x2g_sum(g, rmse * flat_reversed_mask, on_node=on_node)
        mask_sum = x2g_sum(g, flat_reversed_mask, on_node=on_node)
        terms['masked_rmse'] = rmse_masked_sum / torch.clamp(mask_sum, min=1e-6)

        terms['loss'] = terms[terms['loss_type']]

        return terms

    def p_sample(self, model_output, x_t, t, ef_slice, g, on_node, ):
        """
        Sample x_{t-1} from the model at the given timestep.

        Args:
            model_output: has to be x_0_pred for now
            ef_slice: indicis of zero-centered noises
            x_t:
            t:  (dg, 1)  the number of diffusion steps
            g:
            on_node:

        Returns:   a dict containing the following keys:
                 - 'sample': a random sample from the model.

        """

        assert (t > 0).all()

        out = self._p_mean_variance(
            model_output,
            x_t,
            t,
            g,
            on_node,
        )
        noise = self.sample_noise(x_t.shape, x_t.device, ef_slice, g, on_node)
        nonzero_mask = (
            (t != 1).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )  # no noise when t == 1
        nonzero_mask = g2x_copy(g, nonzero_mask, on_node)

        samples = out['mean'] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {'samples': samples, 'mean': out['mean']}

    def _p_mean_variance(
            self, model_output, x_t, t, g, on_node,
    ):
        """

        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x, x_0.

        Args:
            model_output:
            x_t:
            t:
            g:
            on_node:

        Returns: a dict with the following keys:
                     - 'mean': the model mean output.
                     - 'variance': the model variance output.
                     - 'log_variance': the log of 'variance'.
        """
        tm1 = t - 1

        model_variance = _extract_into_tensor(self.posterior_variance, tm1, x_t.shape, g, on_node)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, tm1, x_t.shape, g, on_node)

        if self.model_pred_type == ModelPredType.START_X:
            model_mean, _, _ = self._q_posterior_mean_variance(
                x_start=model_output, x_t=x_t, t=t, g=g, on_node=on_node,
            )
        elif self.model_pred_type == ModelPredType.PREVIOUS_X:
            model_mean = model_output
        else:
            raise NotImplementedError(self.model_pred_type)

        assert model_mean.shape == model_log_variance.shape == x_t.shape

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            # "pred_xstart": pred_xstart,
        }

    def _q_posterior_mean_variance(self, x_start, x_t, t, g, on_node):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        tm1 = t - 1  # minus 1 to get the correct array index
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, tm1, x_t.shape, g, on_node) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, tm1, x_t.shape, g, on_node) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, tm1, x_t.shape, g, on_node)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, tm1, x_t.shape, g, on_node
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


def _extract_into_tensor(arr, timesteps, broadcast_shape, g, on_node):
    """

    map arr[timesteps] ( shape=(dg,1) ) to broadcast_shape (dx,...)

    Args:
        arr:
        timesteps: the real timestep minus 1
        broadcast_shape:
        g:
        on_node:

    Returns:

    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()

    res = g2x_copy(g, res, on_node=on_node)

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
