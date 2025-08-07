# with reference to https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
import einops
import numpy as np
import torch
import torch.nn.functional as F

from lcmg.diffusion.diffusion_base import ModelPredType
from lcmg.utils.g_utils import g2x_copy, x2g_sum


class CategoricalDiffusion:

    def __init__(self, *, class_probs: np.array, betas, model_pred_type: ModelPredType):
        """
        CategoricalDiffusion
        do remember that unlike gaussian diffusion here x_t = a * x_{t-1} + (1-a) * m, not sqrt(a)
        but still, we try to have a consistent SNR rate at a given timestep

        loss_type: CE only
        variance_type: Fixed only

        Args:
            class_probs: marginal distribution of each class  shape=(nx,)
            betas:
            model_pred_type:
        """
        if model_pred_type not in [ModelPredType.START_X, ModelPredType.PREVIOUS_X]:
            raise NotImplementedError()

        self.model_pred_type = model_pred_type

        assert class_probs.ndim == 1
        n_classes = class_probs.shape[0]
        self.class_probs = class_probs
        self.n_classes = n_classes

        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        # Use float64 for accuracy.
        betas = np.concatenate([[0], betas], dtype=np.float64)
        self.betas = betas

        self.num_timesteps = int(betas.shape[0]) - 1

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

    def sample_x_T(self, g, on_node):
        """

        Args:
            g:
            on_node:

        Returns:
            sample (nx, dx)
        """
        n = g.num_nodes() if on_node else g.num_edges()
        marginals = torch.from_numpy(self.class_probs).view(1, -1).float().to(g.device)
        marginals = einops.repeat(marginals, '1 dx -> n dx', n=n)

        samples = self.__sample_from_probs(marginals)

        return samples

    def get_snr(self, t, min=1e-6, max=1e6):
        if (t <= 0).any():
            raise ValueError(f't needs to be greater than 0')

        alpha_cumprod_t = torch.from_numpy(self.alphas_cumprod).to(t.device)[t].float()  # ng, 1
        snr = alpha_cumprod_t / (1 - alpha_cumprod_t)
        return snr.clamp(min=min, max=max)

    def q_sample(self, x_start, t, g, on_node):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        Args:
            x_start: the initial data batch. (nx, dx_0)
            t: number of diffusion steps
            g:
            on_node:

        Returns:
            A noisy version of x_start.
        """
        if (t <= 0).any():
            raise ValueError(f't needs to be greater than 0')

        expanded_t = g2x_copy(g, t, on_node)

        q_t_cumprod = self._get_q_t_cumprod(expanded_t)  # nx, dx_0, dx_t

        x_t_probs = torch.einsum('nij,ni->nj', q_t_cumprod, x_start)  # nx, dx_t

        x_t = self.__sample_from_probs(x_t_probs)

        return x_t


    def q_sample_from_ti(self, x_t, t_init, t_target, g, on_node):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_{t_target} | x_{t_init}).

        Args:
            x_t: the initial data batch. (nx, dx_0)
            t_init: initial t
            t_target: number of diffusion steps
            g:
            on_node:

        Returns:
            A noisy version of x_start.
        """
        if (t_target <= 0).any():
            raise ValueError(f't needs to be greater than 0')

        t_target = g2x_copy(g, t_target, on_node)
        t_init = g2x_copy(g, t_init, on_node)

        q_t_cumprod = self._get_q_t_cumprod_range(t_init, t_target)  # nx, dx_0, dx_t

        x_t_probs = torch.einsum('nij,ni->nj', q_t_cumprod, x_t)  # nx, dx_t

        x_t = self.__sample_from_probs(x_t_probs)

        return {"samples": x_t, "probs": x_t_probs}

    def training_losses(self, *, model_output, x_start, x_t, t, g, on_node: bool, x_mask=None, weight=None,
                        mask_sigma=0):
        """

        Args:
            model_output: shape=(dx, f_dim)
            x_start:  shape=(dx, f_dim)
            x_t: shape=(dx, f_dim)
            t: shape=(dg, 1)
            on_node: is model_output node features or edge features
            x_mask: loss values where x_mask is 0 will be kept
            weight: a manual rescaling weight given to each class.

        Returns:
            a dict that contains following keys:
                - 'loss' shape=(dg, )

        """
        assert model_output.dim() == x_start.dim() == 2

        if (t <= 0).any():
            raise ValueError(f't needs to be greater than 0')

        terms = {}

        terms['loss_type'] = 'masked_ce'
        if self.model_pred_type == ModelPredType.START_X:
            target = x_start

            ce = torch.nn.functional.cross_entropy(model_output, target, reduction='none', weight=weight)
        elif self.model_pred_type == ModelPredType.PREVIOUS_X:
            # target = self._q_posterior_mean_variance(x_start, x_t, t, g, on_node)

            expanded_t = g2x_copy(g, t, on_node)

            q_t = self._get_q_t(expanded_t)  # nx, dx_{t-1}, dx_t
            q_tm1_cumprod = self._get_q_t_cumprod(expanded_t - 1)  # nx, dx_0, dx_{t-1}
            q_x_tm1_given_xt = torch.einsum('nij,nj->ni', q_t, x_t)
            q_x_tm1_given_x0 = torch.einsum('nki,nk->ni', q_tm1_cumprod, x_start)

            q_x_tm1_given_x0_xt = q_x_tm1_given_x0 * q_x_tm1_given_xt

            s = q_x_tm1_given_x0_xt.sum(dim=1, keepdim=True)
            # handling situations when the current x_start or x_t is never considered during train
            q_x_tm1_given_x0_xt[s.squeeze(1) == 0] = 1
            s[s.squeeze(1) == 0] = q_x_tm1_given_x0_xt.size(1)

            q_x_tm1_given_x0_xt = q_x_tm1_given_x0_xt / s

            target = q_x_tm1_given_x0_xt

            ce = torch.nn.functional.cross_entropy(model_output, target, reduction='none', weight=weight)
        else:
            raise NotImplementedError(self.model_pred_type)

        flat_reversed_mask = (~x_mask.bool()).view(-1).float()
        flat_reversed_mask[flat_reversed_mask == 0] = mask_sigma
        ce_masked_sum = x2g_sum(g, ce * flat_reversed_mask, on_node=on_node)
        mask_sum = x2g_sum(g, flat_reversed_mask, on_node=on_node)
        terms['masked_ce'] = ce_masked_sum / torch.clamp(mask_sum, min=1e-6)

        terms['loss'] = terms[terms['loss_type']]

        return terms

    def p_sample(self, model_output, x_t, t, g, on_node):
        """
        Sample x_{t-1} from the model at the given timestep.

        p(x_{t-1}|x_t) = sum_{x_0}( q(x_{t-1}|x_0,x_t)*p(x_0|x_t) )

        Args:
            model_output: (nx, dx_0)
            x_t:  (nx, dx_t)
            t:  (ng, 1)
            g:
            on_node:

        Returns:   a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'probs': the predicted probability distribution of x_{t-1}

        """
        if (t <= 0).any():
            raise ValueError(f't needs to be greater than 0')

        pred_probs = model_output.softmax(-1)

        if self.model_pred_type == ModelPredType.START_X:
            pred_xstart_probs = pred_probs  # ! model output are logits, (nx, dx_0)

            expanded_t = g2x_copy(g, t, on_node)

            q_t = self._get_q_t(expanded_t)  # nx, dx_{t-1}, dx_t
            q_tm1_cumprod = self._get_q_t_cumprod(expanded_t - 1)  # nx, dx_0, dx_{t-1}

            # todo 这里可以做预计算
            tmp = torch.einsum('nij,nj->ni', q_t, x_t)  # for ni in range(nx), tmp = x_t @ q_t.T, nx, dx_{t-1}
            tmp = tmp.unsqueeze(1) * q_tm1_cumprod  # nx, dx_0, dx_{t-1}

            # avoid nan 如果测试集里出现了训练集里没见过的（比如+2电）就会有nan
            # 但是不应该让这种情况出现
            marginals = torch.from_numpy(self.class_probs).view(1, -1).float().to(g.device)
            tmp[tmp.sum(dim=2) == 0] += marginals

            tmp = tmp / (tmp.sum(dim=2, keepdim=True))  # q(x_{t-1} | x_t, x_0), x_0=_onehot(1),_onehot(2),_onehot(3)...

            tmp = tmp * pred_xstart_probs.unsqueeze(2)  # nx, dx_0, dx_{t-1}
            probs = tmp.sum(dim=1)  # p(x_{t-1}|x_t) = sum_{x_0}( q(x_{t-1}|x_0,x_t)*p(x_0|x_t) ), (nx, dx_{t-1})

            ttt = expanded_t.squeeze(1)
            assert ttt.dim() == 1
            probs[ttt == 1] = pred_xstart_probs[ttt == 1]  # if t=1, then t-1=0, use the predicted x_0 directly
        elif self.model_pred_type == ModelPredType.PREVIOUS_X:
            probs = pred_probs
        else:
            raise NotImplementedError(self.model_pred_type)

        assert ((probs.sum(dim=-1) - 1).abs() < 1e-4).all()  # actually here detects nan

        samples = self.__sample_from_probs(probs)

        return {"samples": samples, "probs": probs}

    def _get_q_t(self, t):
        """
        t nx, 1
        """
        beta_t = torch.from_numpy(self.betas).to(t.device)[t].float().unsqueeze(2)  # nx, 1, 1
        q_t = (1 - beta_t) * torch.eye(self.n_classes, device=t.device).unsqueeze(0) + \
              beta_t * torch.from_numpy(self.class_probs).view(1, 1, -1).float().to(t.device)  # (dx,dx)
        return q_t  # (nx, dx, dx)

    def _get_q_t_cumprod(self, t):
        """
        t nx, 1
        """
        alpha_cumprod_t = torch.from_numpy(self.alphas_cumprod).to(t.device)[t].float().unsqueeze(2)  # nx, 1, 1
        q_t_cumprod = alpha_cumprod_t * torch.eye(self.n_classes, device=t.device).unsqueeze(0) + \
                      (1 - alpha_cumprod_t) * torch.from_numpy(self.class_probs).view(1, 1, -1).float().to(
            t.device)  # (dx,dx)
        return q_t_cumprod

    def _get_q_t_cumprod_range(self, t0, t1):
        """
        t nx, 1
        """
        assert (t1>=t0).all()

        device = t1.device
        alphas_cumprod = torch.from_numpy(self.alphas_cumprod).to(device)
        alpha_cumprod_t0 = alphas_cumprod[t0].float().unsqueeze(2)  # nx, 1, 1
        alpha_cumprod_t1 = alphas_cumprod[t1].float().unsqueeze(2)  # nx, 1, 1
        alpha_cumprod_t = alpha_cumprod_t1/alpha_cumprod_t0

        q_t_cumprod = alpha_cumprod_t * torch.eye(self.n_classes, device=device).unsqueeze(0) + \
                      (1 - alpha_cumprod_t) * torch.from_numpy(self.class_probs).view(1, 1, -1).float().to(device)  # (dx,dx)
        return q_t_cumprod

    def __sample_from_probs(self, probs, n_repeats=1, max=False):
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


def test_fn():
    import dgl
    import torch.nn.functional as F
    from lcmg.diffusion.diffusion_base import get_named_beta_schedule

    n_nodes = np.array([1, 2, 3, 4, 5])
    bs = len(n_nodes)
    n_cls = 3
    marginals = np.array([0.5, 0.3, 0.2])
    n_diff_step = 1000

    betas = get_named_beta_schedule('cosine', n_diff_step)
    # print(f'{betas=}')

    t = torch.LongTensor([2, 3, 300, 600, 900]).view(bs, 1) + 1

    g = dgl.batch([dgl.rand_graph(i, 0) for i in n_nodes])

    x_t = F.one_hot(torch.randint(0, n_cls, size=(sum(n_nodes),))).float()
    print(f'{x_t=}')

    pred_x_0 = torch.randn(*x_t.shape).float() * 10  # just logits
    print(f'{pred_x_0=}')

    print('-----------------------------')

    my_model = CategoricalDiffusion(class_probs=marginals,
                                    betas=betas,
                                    model_pred_type=ModelPredType.START_X)

    my_res = my_model.p_sample(pred_x_0,
                               x_t,
                               t + 1,  # to match with midi
                               g=g,
                               on_node=True)['probs']
    print(f'myres\n{my_res}')

    print('-----------------------------')

    from torch.nn.utils.rnn import pad_sequence

    x_t_padded = pad_sequence(x_t.split(g.batch_num_nodes().tolist()),
                              batch_first=True, padding_value=-1)
    pred_x_0_padded = pad_sequence(pred_x_0.split(g.batch_num_nodes().tolist()),
                                   batch_first=True, padding_value=-1)
    padding_mask = x_t_padded != -1

    class A:
        pass

    z_t = A()
    z_t.X = x_t_padded
    z_t.node_mask = padding_mask
    z_t.t_int = t
    z_t.pos = torch.zeros(*x_t_padded.shape[:-1], 3)
    max_n = max(n_nodes)
    z_t.E = torch.arange(bs * max_n * max_n * n_cls, dtype=torch.float).view(bs, max_n, max_n, n_cls)
    z_t.charges = z_t.X

    pred = A()
    pred.X = pred_x_0_padded
    pred.E = z_t.E + torch.rand_like(z_t.E)
    pred.charges = pred.X
    pred.pos = pred.X

    cfg = A()
    cfg.model = A()
    cfg.model.diffusion_steps = n_diff_step - 1  #

    from midi.diffusion.diffusion_utils import cosine_beta_schedule_discrete

    cfg.betas = cosine_beta_schedule_discrete(n_diff_step - 1, [1, 1, 1, 1, 1])

    from midi.diffusion.noise_model import MarginalUniformTransition

    marginals_tensor = torch.from_numpy(marginals)
    digress_model = MarginalUniformTransition(cfg, marginals_tensor, marginals_tensor, marginals_tensor, 2)

    tm1 = t - 1
    midi_ref_res = digress_model.sample_zs_from_zt_and_pred(z_t, pred, tm1)[padding_mask].reshape(my_res.shape)

    # midi_ref_res = torch.FloatTensor([[[0.2126, 0.3509, 0.4365],  # midi当输入的t为1的时候，输出其实是x_2的估计
    #                                    [0.2843, 0.4158, 0.2999],
    #                                    [0.3075, 0.3609, 0.3316],
    #                                    [0.3300, 0.3423, 0.3276],
    #                                    [0.2960, 0.4294, 0.2746]],
    #
    #                                   [[0.1520, 0.3704, 0.4777],
    #                                    [0.2949, 0.4159, 0.2892],
    #                                    [0.3180, 0.3662, 0.3158],
    #                                    [0.3551, 0.3354, 0.3095],
    #                                    [0.3283, 0.4054, 0.2664]],
    #
    #                                   [[0.0976, 0.3923, 0.5101],
    #                                    [0.3663, 0.3908, 0.2429],
    #                                    [0.3845, 0.3666, 0.2488],
    #                                    [0.4499, 0.3121, 0.2380],
    #                                    [0.4456, 0.3240, 0.2304]],
    #
    #                                   [[0.0999, 0.4027, 0.4974],
    #                                    [0.3874, 0.3807, 0.2319],
    #                                    [0.4037, 0.3635, 0.2328],
    #                                    [0.4709, 0.3075, 0.2217],
    #                                    [0.4708, 0.3076, 0.2216]]])

    print(f'midi_ref_res(ahead by 1 step)\n{midi_ref_res}')
    print('-----------------------')

    print(f'{my_res - midi_ref_res}')


if __name__ == '__main__':
    test_fn()
