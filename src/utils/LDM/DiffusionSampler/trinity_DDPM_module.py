"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
from typing import Any, Optional, Union, List
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from functools import partial
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from models.trinityLDM.util import exists, instantiate_from_config
from models.trinityLDM.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from models.trinityLDM.util import default
from models.trinityLDM.modules.ema import LitEma
from src import utils
from tqdm import tqdm
from src.models.modules import GaussianDiffusion, DiffusionOutput, MeanScaler, \
    NOPScaler
from src.models.modules.distribution_output import GaussianDiag

from einops import rearrange, repeat
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

log = utils.get_pylogger(__name__)


class TrinityDDPM(pl.LightningModule):
    def __init__(self,
                 # region init parameters
                 given_betas=None,
                 beta_schedule='linear',
                 timesteps=1000,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 parameterization="eps",
                 cell_type: str = 'LSTM',  # LSTM / GRU
                 input_size=927,
                 hidden_size=512,
                 num_layers=2,
                 dropout=0.2,
                 epsilon_theta_model=None,
                 loss_type='l2',
                 learning_rate=1.0e-04,
                 max_lr=0.2,
                 learn_logvar=False,
                 cond_stage_trainable=False,
                 logvar_init=0.,
                 l_simple_weight=1.,
                 original_elbo_weight=0.,
                 scheduler_config=None,
                 use_ema=True,
                 init_by_mean_pose=False,
                 num_parallel_samples=1,
                 image_size=1,
                 channels=45,
                 log_every_t=100,
                 clip_denoised=True,
                 quantile=0.5,
                 bvh_save_path: str = "",
                 sampler='DDPM',
                 ddim_steps=200,
                 solver_steps=5,
                 solver_order=1,
                 solver_skip_type='time_uniform',  # 'time_uniform' or 'logSNR' or 'time_quadratic'
                 solver_method='multistep',  # 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'
                 # endregion
                 ):
        super().__init__()
        # self.save_hyperparameters(ignore=['epsilon_theta_model'])
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.batch = None
        self.n_timesteps = None
        self.n_feats = None
        self.n_lookahead = None
        self.seqlen = None
        self.inited_rnn = None
        self.linear_start = None
        self.num_timesteps = None
        self.beta_schedule = beta_schedule
        self.state = None
        self.bvh_save_path = bvh_save_path
        self.save_hyperparameters(logger=False)
        self.v_posterior = v_posterior
        self.timesteps = timesteps
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.cell_type = cell_type
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            # bidirectional=True,

        )
        self.save_hyperparameters(logger=False)
        # self.save_hyperparameters(ignore=['epsilon_theta_model'])
        self.model = epsilon_theta_model  # customized model
        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.learn_logvar = learn_logvar
        self.cond_stage_trainable = cond_stage_trainable
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        self.loss_type = loss_type
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.num_parallel_samples = num_parallel_samples
        self.init_by_mean_pose = init_by_mean_pose
        self.image_size = image_size
        self.channels = channels
        self.log_every_t = log_every_t
        self.clip_denoised = clip_denoised
        self.quantile = quantile
        self.distr_output = DiffusionOutput(
            self, input_size=channels, cond_size=hidden_size
        )
        self.proj_dist_args = self.distr_output.get_args_proj(512)
        self.normal_distribution = GaussianDiag()
        self.sampler = sampler
        self.ddim_steps = ddim_steps
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self.solver_steps = solver_steps
        self.solver_order = solver_order
        self.solver_skip_type = solver_skip_type  # 'time_uniform' or 'logSNR' or 'time_quadratic'
        self.solver_method = solver_method  # 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)  # β
        alphas = 1. - betas  # α
        alphas_cumprod = np.cumprod(alphas, axis=0)  #
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def prepare_cond(self, jt_data, ctrl_data):
        # log.info(f'type of jt_data {jt_data.device} ,ctrl_data : {ctrl_data.device}')
        jt_data = jt_data.cuda()
        ctrl_data = ctrl_data.cuda()
        # log.info(f'to cuda type of jt_data {jt_data.device} ,ctrl_data : {ctrl_data.device}')
        nn, seqlen, n_feats = jt_data.shape
        # log.info('encode_cond........')
        jt_data = torch.reshape(jt_data, (nn, seqlen * n_feats))  # jt_data [80,225]
        nn, seqlen, n_feats = ctrl_data.shape
        ctrl_data = torch.reshape(ctrl_data, (nn, seqlen * n_feats))  # ctrl_data [80,702]
        # log.info(f'jt_data shape: {jt_data.shape}, ctrl_data shape: {ctrl_data.shape}')
        # #jt_data [80,225]  ctrl_data [80,702]
        cond = torch.cat((jt_data, ctrl_data), 1)  # [80,927]
        cond = torch.unsqueeze(cond, 1)  # [80,1,927]

        return cond

    def repeat_tensor(self, tensor, dim=0):
        return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

    def prepare_data_for_test(self, batch: Any, batch_idx: int):
        autoreg_all = batch["autoreg"].cuda()  # [64, 400, 45]
        # log.info(f'test_step -> autoreg_all shape: {autoreg_all.shape} \n {autoreg_all}')
        control_all = batch["control"].cuda()  # [64,400,27]
        self.seqlen = self.trainer.datamodule.seqlen
        self.n_lookahead = self.trainer.datamodule.n_lookahead
        self.batch, self.n_timesteps, self.n_feats = autoreg_all.shape
        sampled_all = torch.zeros((self.batch, self.n_timesteps - self.n_lookahead, self.n_feats)).cuda()  # [80,380,45]
        if self.init_by_mean_pose:
            autoreg = torch.zeros((self.batch, self.seqlen, self.n_feats), dtype=torch.float32).cuda()  # [80,5,45]
        else:
            autoreg = autoreg_all[:, :self.seqlen, :]
        control_all = control_all.cuda()
        sampled_all[:, :self.seqlen, :] = autoreg  # start pose [0,0,0,0,0]

        log.info('Sampling_decoder')
        future_samples = sampled_all

        repeated_states = None

        control = control_all[:, 0:(self.seqlen + 1 + self.n_lookahead), :]
        combined_cond = self.prepare_cond(autoreg, control)
        return future_samples, control_all, combined_cond, autoreg

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            # self.model_ema.store(self.model.parameters())
            # self.model_ema.copy_to(self.model)
            if context is not None:
                # log.info(f"{context}: Switched to EMA weights")
                pass
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    # log.info(f"{context}: Restored training weights")
                    pass

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def distr_args(self, rnn_outputs: torch.Tensor):
        """
        Returns the distribution of DeepVAR with respect to the RNN outputs.

        Parameters
        ----------
        rnn_outputs
            Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)

        Returns
        -------
        distr
            Distribution instance
        distr_args
            Distribution arguments
        """
        (distr_args,) = self.proj_dist_args(rnn_outputs)

        # # compute likelihood of target given the predicted parameters
        # distr = self.distr_output.distribution(distr_args, scale=scale)

        # return distr, distr_args
        return distr_args

    def q_sample(self, x_start, t, noise=None):
        '''
        This is the forward process step: sqrt(alpha_cumprod)* x_0 + sqrt(1 - alpha_cumprod) * noise

        Args:
            x_start: x_0
            t: timestep
            noise: standard Gaussian distribution

        Returns: sqrt(alpha_cumprod)* x_0 + sqrt(1 - alpha_cumprod) * noise

        '''
        noise = default(noise, lambda: torch.randn_like(x_start))  # [6080,1,45]
        x_t = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
               extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        return x_t

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        elif self.loss_type == "huber":
            loss = torch.nn.functional.smooth_l1_loss(pred, target)
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, cond, timestep, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))  # [6080,1,45]
        x_t = self.q_sample(x_start=x_start, t=timestep, noise=noise)  # [6080,1,45]
        model_out = self.model(x_t, cond, timestep)  # cond_inplanes [6080,1,45] t [6080]  model_out [6080,1,45]

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False)  # [6080,1,45]
        # loss = loss.mean(dim=[1, 2, 3])
        loss = loss.mean(dim=[1, 2])  # [6080]

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[timestep] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, cond, t, clip_denoised: bool):
        # model_out = self.model(x, t)
        model_out = self.model(x, cond, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=True, repeat_noise=False):
        '''
        This is the reverse step(Denoise step)
        '''
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(cond, (batch_size, image_size, channels),
                                  return_intermediates=return_intermediates)

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False):
        '''
        This is the denoising process:
        Args:
            cond:
            shape:
            return_intermediates:

        Returns:

        '''
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]

        for i in tqdm(reversed(range(0, self.num_timesteps)),
                      total=self.num_timesteps, leave=False, position=1, desc='Sample T'
                      ):
            img = self.p_sample(x=img, cond=cond,
                                t=torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates

        return img

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=True, repeat_noise=False):
        '''
        This is the reverse step(Denoise step)
        '''
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        x_recon = self.model(x_noisy, cond, t)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def forward(self, x: torch.Tensor, cond: torch.Tensor, *args, **kwargs) -> Any:
        # self.device = x.device

        rnn_outputs, _ = self.rnn(cond)  # [64,95,512]
        distr_args = self.distr_args(rnn_outputs=rnn_outputs)  # [64,95,512]
        B, T, _ = x.shape
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        t = torch.randint(0, self.num_timesteps, (B * T,), device=self.device).long()
        return self.p_losses(x.reshape(B * T, 1, -1), distr_args.reshape(B * T, 1, -1), t, *args, **kwargs)

    '''Train Step'''

    def on_train_start(self) -> None:
        pass

    def on_train_epoch_start(self) -> None:
        pass

    def model_step(self, batch: Any):
        x = batch["x"]  # [64, 95, 65]  the output of body pose corresponding to the condition [80,95,45]
        cond = batch["cond_inplanes"]  # [64, 95, 2405]
        loss, loss_dict = self.forward(x, cond)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.model_step(batch)
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", float(self.global_step),
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().optimizer.param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # if self.global_step % 50 == 0:
        #     log.info(f'train loss: {loss}')
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        pass

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    '''Validataion Step'''

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        loss, loss_dict_no_ema = self.model_step(batch)
        with self.ema_scope():
            loss, loss_dict_ema = self.model_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # if self.global_step % 50 == 0:
        #     log.info(f'val loss: {loss}')

    @torch.no_grad()
    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        pass

    '''Test Step'''

    @torch.no_grad()
    def on_test_start(self) -> None:
        log.info("on_test_start")

    @torch.no_grad()
    def on_test_epoch_start(self) -> None:
        log.info("on_test_epoch_start")

    @torch.no_grad()
    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        log.info("test_step")
        future_samples, control_all, combined_cond, autoreg = self.prepare_data_for_test(batch, batch_idx)

        rnn_outputs, self.state = self.rnn(combined_cond)
        if self.cell_type == "LSTM":
            repeated_states = [self.repeat_tensor(s, dim=1) for s in self.state]
        else:
            repeated_states = self.repeat_tensor(self.state, dim=1)
        self.inited_rnn = True

        repeated_control_all = self.repeat_tensor(control_all)
        # repeated_autoreg = repeat_tensor(autoreg)
        repeated_future_samples = self.repeat_tensor(future_samples)

        for k in tqdm(range(control_all.shape[1] - self.seqlen - self.n_lookahead - 1), colour='yellow',
                      leave=True, position=0, desc='Frame'):
            repeated_control = repeated_control_all[:, (k + 1):((k + 1) + self.seqlen + 1 + self.n_lookahead), :]
            repeated_autoreg = self.repeat_tensor(autoreg)
            combined_cond = self.prepare_cond(repeated_autoreg, repeated_control)
            rnn_outputs, repeated_states = self.rnn(combined_cond, repeated_states)
            distr_args = self.distr_args(rnn_outputs=rnn_outputs)
            # sample
            with self.ema_scope("Plotting"):
                if self.sampler == "DDPM":
                    samples = self.sample(cond=distr_args, batch_size=repeated_control.shape[0],
                                          return_intermediates=False)
                elif self.sampler == "DDIM":
                    from src.models.TrinityDiffusion.Trinity_ddim import TrinityDDIMSampler
                    ddim_sampler = TrinityDDIMSampler(self, schedule=self.beta_schedule)
                    shape = (self.image_size, self.channels,)
                    samples, intermediates = ddim_sampler.sample(S=self.ddim_steps,
                                                                 batch_size=repeated_control.shape[0],
                                                                 shape=shape, cond=distr_args, verbose=False)
                elif self.sampler in ['dpmsolver', 'dpmsolver++']:
                    from TrinityDiffusion.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
                    model_kwargs = {"cond_inplanes": distr_args}
                    noise_schedule = NoiseScheduleVP(schedule='cosine')  # , betas=self.betas
                    model_fn = model_wrapper(
                        self.model,
                        noise_schedule,
                        model_type="noise",  # or "x_start" or "v" or "score"
                        model_kwargs=model_kwargs,
                    )
                    shape = (repeated_control.shape[0], self.image_size, self.channels,)  # [64,1,45]
                    img = torch.randn(shape, device=self.device)
                    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    samples = dpm_solver.sample(img, steps=self.solver_steps,  order=self.solver_order, t_start=self.linear_start, t_end=self.linear_end,
                                                      skip_type='time_uniform', method='multistep')
                    # dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    # samples = dpm_solver.sample(img, steps=self.solver_steps,  t_end=1e-3, order=3,  t_start=2e-3,
                    #                                   skip_type='time_uniform', method='singlestep')
                    # dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type=self.sampler)
                    # samples = dpm_solver.sample(
                    #     img,
                    #     steps=self.solver_steps,
                    #     order=self.solver_order,
                    #     skip_type=self.solver_skip_type,  # "time_uniform",
                    #     method=self.solver_method,  # "singlestep",
                    # )

            samples = samples[:, 0, :]
            samples = samples.reshape(-1, self.num_parallel_samples, self.n_feats)
            quantile_new_samples = torch.quantile(samples, self.quantile, dim=1)
            future_samples[:, (k + self.seqlen), :] = quantile_new_samples
            autoreg = torch.cat((autoreg[:, 1:, :], quantile_new_samples[:, None, :]), dim=1)
        future_samples = future_samples.cpu().numpy().copy()
        datamodule = self.trainer.datamodule
        datamodule.save_animation(motion_data=future_samples, filename=self.bvh_save_path,
                                  paramValue="diffstep_" + str(self.timesteps))
        return samples

    @torch.no_grad()
    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        log.info("test_epoch_end")

    '''Predict Step'''

    @torch.no_grad()
    def on_predict_start(self) -> None:
        pass

    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pass

    @torch.no_grad()
    def on_predict_epoch_end(self, results: List[Any]) -> None:
        pass

    def configure_optimizers_stable_diffusion(self):
        lr = self.learning_rate

        params = list(self.model.parameters())
        params = params + list(self.rnn.parameters())
        params = params + list(self.proj_dist_args.parameters())

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.parameters())
        # params = list(self.model.parameters())
        # params = params + list(self.rnn.parameters())
        # params = params + list(self.proj_dist_args.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)
        lr_scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            verbose=False
        )
        # lr_scheduler = ReduceLROnPlateau(
        #     optimizer=optimizer,
        #     mode='min',
        #     min_lr=1e-5
        # )
        scheduler = [
            {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1
            }]
        # lr_scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=self.max_lr,
        #     pct_start=0.3,
        #     anneal_strategy='cos',
        #     cycle_momentum=True,
        #     base_momentum=0.85,
        #     max_momentum=0.95,
        #     div_factor=25.0,
        #     final_div_factor=10000.0,
        #     three_phase=False,
        #     last_epoch=-1,
        #     verbose=True,
        #     steps_per_epoch=self.trainer.datamodule.batch_size,
        #     epochs=self.trainer.max_epochs,
        # )
        # optimizer = Adam(
        #     self.model.parameters(), lr=self.learning_rate, weight_decay=1e-6
        # )
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        # lr_scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=self.max_lr,
        #     steps_per_epoch=self.trainer.datamodule.batch_size,
        #     epochs=self.trainer.max_epochs,
        # )
        return [optimizer], scheduler

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     scheduler.step(epoch=self.current_epoch)
    # def configure_optimizers(self) -> Any:
    #     lr = self.learning_rate
    #     params = list(self.model.parameters())
    #     if self.learn_logvar:
    #         params = params + [self.logvar]
    #     opt = torch.optim.AdamW(params, lr=lr)
    #     return opt

    # if __name__ == "__main__":
