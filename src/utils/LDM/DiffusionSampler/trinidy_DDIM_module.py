from TrinityDiffusion.trinity_DDPM_module import TrinityDDPM
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from typing import Any, Optional, Union, List
import torch
import torch.nn as nn
import numpy as np
from src import utils
from tqdm import tqdm
from src.models.ldm.models.diffusion.ddim import DDIMSampler

log = utils.get_pylogger(__name__)


class TrinityDDIM(TrinityDDPM):
    def __init__(self,
                 ddim_steps: int = 200,
                 ddim_eta=1.0, ):
        super().__init__()
        self.ddim_steps = ddim_steps
        self.ddim_eta = 1.0

    def training_step(self, batch, batch_idx):
        super().training_step()

    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        super().validation_step()

    @torch.no_grad()
    def on_test_start(self) -> None:
        log.info("on_test_start")

    @torch.no_grad()
    def on_test_epoch_start(self) -> None:
        log.info("on_test_epoch_start")

    @torch.no_grad()
    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        log.info("test_step")
        autoreg_all = batch["autoreg"].cuda()  # [20, 400, 45]
        # log.info(f'test_step -> autoreg_all shape: {autoreg_all.shape} \n {autoreg_all}')
        control_all = batch["control"].cuda()  # [80,400,27]
        seqlen = self.trainer.datamodule.seqlen
        n_lookahead = self.trainer.datamodule.n_lookahead
        batch, n_timesteps, n_feats = autoreg_all.shape
        sampled_all = torch.zeros((batch, n_timesteps - n_lookahead, n_feats)).cuda()  # [80,380,45]
        autoreg = torch.zeros((batch, seqlen, n_feats), dtype=torch.float32).cuda()  # [80,5,45]
        control_all = control_all.cuda()
        sampled_all[:, :seqlen, :] = autoreg  # start pose [0,0,0,0,0]

        log.info('Sampling_decoder')
        future_samples = sampled_all

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        repeated_states = None

        control = control_all[:, 0:(seqlen + 1 + n_lookahead), :]
        combined_cond = self.prepare_cond(autoreg, control)

        rnn_outputs, self.state = self.rnn(combined_cond)
        if self.cell_type == "LSTM":
            repeated_states = [repeat(s, dim=1) for s in self.state]
        else:
            repeated_states = repeat(self.state, dim=1)
        self.inited_rnn = True

        repeated_control_all = repeat(control_all)
        # repeated_autoreg = repeat(autoreg)
        repeated_future_samples = repeat(future_samples)

        for k in tqdm(range(control_all.shape[1] - seqlen - n_lookahead - 1), colour='yellow', position=0,
                      leave=True, ):
            repeated_control = repeated_control_all[:, (k + 1):((k + 1) + seqlen + 1 + n_lookahead), :]
            repeated_autoreg = repeat(autoreg)
            combined_cond = self.prepare_cond(repeated_autoreg, repeated_control)
            rnn_outputs, repeated_states = self.rnn(combined_cond, repeated_states)
            # sample
            use_ddim = self.ddim_steps is not None
            with self.ema_scope("Plotting"):
                if use_ddim:
                    ddim_sampler = DDIMSampler(self)
                    shape = (64, 1, self.channels)
                    samples = ddim_sampler.sample(ddim_steps=self.ddim_steps, batch_size=64, shape=shape, )
                    # samples = self.sample(cond=rnn_outputs, batch_size=repeated_control.shape[0],
                    #                       return_intermediates=False)
            samples = samples[:, 0, :]
            samples = samples.reshape(-1, self.num_parallel_samples, n_feats)
            quantile_new_samples = torch.quantile(samples, self.quantile, dim=1)
            future_samples[:, (k + seqlen), :] = quantile_new_samples
            autoreg = torch.cat((autoreg[:, 1:, :], quantile_new_samples[:, None, :]), dim=1)
        future_samples = future_samples.cpu().numpy().copy()
        datamodule = self.trainer.datamodule
        datamodule.save_animation(motion_data=future_samples, filename=self.bvh_save_path,
                                  paramValue="diffstep_" + str(self.timesteps))
        return samples
