"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
from typing import Any, Optional, Union, List
from lightning.pytorch.utilities.types import STEP_OUTPUT
from functools import partial
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR
from lightning import LightningModule, LightningDataModule
import torch
import torch.nn as nn
import numpy as np
from src.utils.LDM.util import exists, instantiate_from_config
from src.utils.LDM.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from src.utils.LDM.util import default
from src.utils.LDM.modules.ema import LitEma
from src import utils
from tqdm import tqdm
import os
import time
from lightning.pytorch.profilers import SimpleProfiler, PassThroughProfiler
import gc

# import hydra
log = utils.get_pylogger(__name__)


class TrinityDiffmotionModule(LightningModule):
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
                 # hidden_size=512,
                 gesture_features=67,
                 dropout=0.2,
                 encoder_dim=1024,
                 eps_theta_mod=None,
                 loss_type='l2',
                 learning_rate=1.0e-04,
                 learn_logvar=False,  # Need to test
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
                 bvh_save_file: str = "",
                 batch_smooth: bool = False,
                 concate_length: int = 5,
                 sampler='DDPM',
                 ddim_steps=200,
                 solver_steps=5,
                 solver_order=1,
                 solver_skip_type='time_uniform',  # 'time_uniform' or 'logSNR' or 'time_quadratic'
                 solver_method='multistep',  # 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'
                 num_sequences=3,
                 use_wavlm=True,
                 param_for_name="",
                 matmul_precision='high',
                 log_sync_dist=False,
                 # profiler=None,
                 # endregion
                 ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['eps_theta_mod'])
        # self.save_hyperparameters(logger=False)
        # also ensures init params will be stored in ckpt
        torch.set_float32_matmul_precision(matmul_precision)
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
        self.gesture_features = gesture_features
        self.bvh_save_path = bvh_save_path
        self.bvh_save_file = bvh_save_file
        if isinstance(param_for_name, list):
            self.param_for_name = '_' + param_for_name[0]
        elif isinstance(param_for_name, str):
            self.param_for_name = '_' + param_for_name
        self.batch_smooth = batch_smooth
        self.concate_length = concate_length

        self.v_posterior = v_posterior
        self.timesteps = timesteps
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.model = eps_theta_mod  # customized model
        self.learning_rate = learning_rate
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
        self.num_sequences = num_sequences
        self.encoder_dim = encoder_dim
        self.use_wavlm = use_wavlm
        self.log_sync_dist = log_sync_dist
        self.process_state = "training"
        # self.profiler = profiler or PassThroughProfiler()

        # print("")

    # region Diffusion Block--------------------------------------------------------
    ######################################Diffusion Block#################################################################
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

    def repeat_tensor(self, tensor, dim=0):
        return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

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

    def q_sample(self, x_start, t, noise=None):
        """
        This is the forward process step: sqrt(alpha_cumprod)* x_0 + sqrt(1 - alpha_cumprod) * noise
        Args:
            x_start: x_0
            t: timestep
            noise: standard Gaussian distribution

        Returns: sqrt(alpha_cumprod)* x_0 + sqrt(1 - alpha_cumprod) * noise

        """
        noise = default(noise, lambda: torch.randn_like(x_start))  # [B, 360, 65]
        x_t = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
               extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)  # [64, 360, 65]
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
        noise = default(noise, lambda: torch.randn_like(x_start))  # [B, 120, 65]
        x_t = self.q_sample(x_start=x_start, t=timestep, noise=noise)  # [B, 120, 65]
        model_out = self.model(inputs=x_t, cond=cond,
                               time=timestep, processing_state=self.process_state,
                               last_time_stamp=self.num_timesteps)  # x_t: [B, 120, 65]  cond: [B, 120, 27]  time: [32]            # cond_inplanes [6080,1,45] t [6080]  model_out [6080,1,45]

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False)  # [320,120,65]
        # loss = loss.mean(dim=[1, 2, 3])
        loss = loss.mean(dim=[1, 2])  # [B,]

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
        model_out = self.model(inputs=x, cond=cond, time=t, processing_state=self.process_state, last_time_stamp=self.timesteps-1)  # [B, 400, 65]
        # model_out = self.model(x, cond, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
        '''
        This is the reverse step(Denoise step)
        '''
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        if self.batch_smooth:
            for i in torch.arange(1, noise.shape[0]):
                i = torch.tensor(i).type(torch.long)
                noise[i, :self.concate_length, :] = noise[i - 1, -self.concate_length:, :]
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, cond, shape, return_intermediates=False):  # cond [16,1000, 768]
        return self.p_sample_loop(cond, shape=shape,
                                  return_intermediates=return_intermediates)

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)  # [B, 400, 65]
        if return_intermediates:
            intermediates = [img]
        if self.batch_smooth:
            for i in torch.arange(1, img.shape[0]):
                i = torch.tensor(i).type(torch.long)
                img[i, :self.concate_length, :] = img[i - 1, -self.concate_length:, :]

        for i in tqdm(reversed(range(0, self.num_timesteps)),
                      total=self.num_timesteps, leave=False, position=1, desc='Sample T'
                      ):
            # self.log("Inference step", i)
            self.log("Inference_step", float(i),
                     prog_bar=True, logger=True, on_step=True, on_epoch=False)
            img = self.p_sample(x=img, cond=cond,
                                t=torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if return_intermediates:
                if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                    intermediates.append(img)
        if return_intermediates:
            return img, intermediates

        return img

    ##################################################################################################################
    # endregion Diffusion Block

    def model_step(self, batch: Any):
        joint_data = batch['joint_data']  # [B, T, Feature] [B,360,65]
        cond = batch['cond']  # [B, T, Feature] [B, 360, 80]
        loss, loss_dict = self.forward(joint_data, cond)
        if torch.isnan(loss).any():
            raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        x_recon = self.model(x_noisy, cond, t, processing_state=self.process_state,
                             last_time_stamp=self.num_timesteps)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def forward(self, x: torch.Tensor, cond: torch.Tensor, *args, **kwargs) -> Any:
        B, T, F = x.shape  # 64, 360, 64   # [B, T, F]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()  # [B,] the noise step index

        return self.p_losses(x, cond, t, *args, **kwargs)

    '''Train Step'''

    def on_train_start(self) -> None:
        self.process_state = "training"
        # log.info(f'Processing state: {self.process_state}')

    def on_train_epoch_start(self) -> None:
        pass

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.model_step(batch)
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True, sync_dist=self.log_sync_dist)

        self.log("global_step", float(self.global_step),
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().optimizer.param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        log.info(f'{self.trainer.current_epoch} Done!')
        pass

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    '''Validataion Step'''

    def on_validation_start(self) -> None:
        self.process_state = "validation"
        # log.info(f'Processing state: {self.process_state}')

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        loss, loss_dict_no_ema = self.model_step(batch)
        with self.ema_scope():
            loss, loss_dict_ema = self.model_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                      sync_dist=self.log_sync_dist)
        self.log_dict(loss_dict_ema, prog_bar=True, logger=True, on_step=False, on_epoch=True,
                      sync_dist=self.log_sync_dist)

    '''Test Step'''

    @torch.no_grad()
    def on_test_start(self) -> None:
        self.process_state = "test"
        # log.info(f'Processing state: {self.process_state}')

    @torch.no_grad()
    def on_test_epoch_start(self) -> None:
        log.info("on_test_epoch_start")

    @torch.no_grad()
    def prepare_data_for_test(self, batch: Any, batch_idx: int, data_module=None):
        control_all = batch["cond"]  # [B, 400, 27]
        if self.use_wavlm:
            self.batch, self.n_timesteps = control_all.shape
            self.n_timesteps = self.n_timesteps // data_module.audio_sample_rate * data_module.framerate
        else:
            self.batch, self.n_timesteps, self.features = control_all.shape
        sampled_all = torch.zeros((self.batch, self.n_timesteps, self.gesture_features))  # [80,380,45]
        log.info('Sampling_decoder')
        return sampled_all, control_all

    @torch.no_grad()
    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        data_module = self.trainer.datamodule

        future_samples, control_all = self.prepare_data_for_test(batch, batch_idx,
                                                                 data_module=data_module)  # [B, 400, 65] [B, 400, 27]

        for num in range(0, self.num_sequences):
            start_time = time.perf_counter()
            log.info(f'Generating {num} sequences: file name: {self.bvh_save_file}!!!')
            with self.ema_scope("Plotting"):
                if self.sampler == "DDPM":
                    samples = self.sample(cond=control_all,
                                          shape=future_samples.shape,
                                          # batch_size=control_all.shape[0],
                                          return_intermediates=False)
                elif self.sampler == "DDIM":
                    pass
                    # from src.diffmotion.components.samplers.Trinity_ddim import TrinityDDIMSampler
                    # ddim_sampler = TrinityDDIMSampler(model=self, schedule=self.beta_schedule)
                    # shape = (control_all.shape[1] // 16000 * 20, self.gesture_features)
                    # samples, intermediates = ddim_sampler.sample(S=self.ddim_steps,
                    #                                              batch_size=control_all.shape[0],
                    #                                              shape=shape,
                    #                                              cond=control_all,
                    #                                              verbose=False)
                elif self.sampler in ['dpmsolver', 'dpmsolver++']:
                    from src.utils.LDM.DiffusionSampler.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, \
                        DPM_Solver
                    # 1. 定义噪声进度 (VP 指的是 Variance Preserving，即 DDPM/DDIM 的基础)
                    # 使用与你代码一致的 alphas_cumprod
                    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

                    # 2. 封装模型，使其适配 Solver 接口
                    model_kwargs = {"cond": control_all}  # 对应你 model 的 cond 参数
                    model_fn = model_wrapper(
                        self.model,
                        noise_schedule,
                        model_type="noise",  # 你模型预测的是 eps
                        model_kwargs=model_kwargs,
                    )

                    # 3. 初始化 Solver
                    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type=self.sampler)

                    # 4. 执行采样
                    # S 对应 solver_steps，通常 10-20 步即可
                    samples = dpm_solver.sample(
                        future_samples.to(self.device),  # 初始噪声
                        steps=self.solver_steps,
                        order=self.solver_order,
                        skip_type=self.solver_skip_type,
                        method=self.solver_method,
                    )
            # TODO
            # if use_gesture_encoder:
            #    future_samples = Decoder(future_samples)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            self.log(f'Seq_{num}_Generate_time', execution_time)
            # log.info(f'Seq_{num}_Generate_time', execution_time)
            future_samples = samples.cpu().numpy().copy()
            # path = os.path.dirname(self.bvh_save_path)

            bvh_save_name = os.path.join(self.bvh_save_path, self.bvh_save_file)
            extract_parameters = utils.extract_characters(self.bvh_save_path)
            for parameter in extract_parameters:
                if '.yaml' not in parameter and '.ckpt' not in parameter:
                    bvh_save_name = bvh_save_name + "_" + parameter
            log.info(f'bvh_save_name: {bvh_save_name}')

            # self.param_for_name += str(batch_idx)
            self.trainer.datamodule.save_animation(motion_data=future_samples, filename=bvh_save_name,
                                                   paramValue=self.param_for_name + "_" + str(num))
        return samples

    '''Predict Step'''
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
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
