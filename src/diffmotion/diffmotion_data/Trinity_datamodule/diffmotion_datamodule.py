from typing import Optional
from lightning import LightningDataModule
import joblib as jl
import os
from src import utils
import numpy as np
from src.diffmotion.diffmotion_data.Trinity_datamodule.trinity_motion_data import MotionDataset, TestDataset
from torch.utils.data import DataLoader, Dataset
from src import utils
from src.utils.pymo.writers import *
# from src.data.pymo.writers import *
import torch
from scipy import stats
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plot
from textwrap import wrap
# from prefetch_generator import BackgroundGenerator
import torch.nn as nn

log = utils.get_pylogger(__name__)


# class DataLoaderX(DataLoader):
#
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())


def standardize(data, scaler):
    shape = data.shape
    print(f"standarize data.shape: {data.shape}")
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))  # StandardScaler expected <= 2
    scaled = scaler.transform(flat).reshape(shape)
    return scaled


def inv_standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled


class GestureDataModule(LightningDataModule):
    def __init__(self,
                 data_root: str = "data/TrinityDataSet/processed",
                 framerate: str = 20,
                 seqlen: int = 5,
                 n_lookahead: int = 20,
                 dropout: float = 0.4,
                 batch_size: int = 32,
                 audio_sample_rate: int = 16000,
                 pin_memory: bool = False,
                 gesture_features: int = 65,
                 input_size: int = 972,
                 num_workers: int = 16,
                 is_full_body: bool = False,
                 is_smoothing: bool = True,
                 window_length: int = 51,
                 polyorder: int = 2,
                 save_np: bool = False,
                 test_style: str = '',
                 ):
        super(GestureDataModule, self).__init__()
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.data_root = data_root
        self.framerate = framerate
        self.seqlen = seqlen
        self.n_lookahead = n_lookahead
        self.dropout = dropout
        self.batch_size = batch_size
        self.audio_sample_rate = audio_sample_rate
        self.pin_memory = pin_memory
        self.input_size = input_size
        self.num_workers = num_workers
        self.is_full_body = is_full_body
        self.is_smoothing = is_smoothing
        self.window_length = window_length
        self.polyorder = polyorder
        self.n_x_channels = None
        self.n_test = None
        self.test_output = None
        self.test_input = None
        self.val_output = None
        self.val_input = None
        self.train_output = None
        self.train_input = None
        self.data_pipe = None
        self.output_scaler = None
        self.input_scaler = None
        self.save_np = save_np
        self.test_style = test_style
        self.gesture_features = gesture_features
        # self.save_hyperparameters(logger=False)
        scaler_dir = os.path.join(self.data_root, 'input_scaler.sav')
        self.is_inited_with_std = os.path.exists(scaler_dir)
        self.train_dataset: Optional[Dataset] = None
        self.validation_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        if 'waveform' in self.data_root:
            self.is_waveform = True
        else:
            self.is_waveform = False
        log.info(f'Is using scaler: {self.is_inited_with_std}----------')
        log.info(f'Data root path: {self.data_root}--------------------')

    def prepare_data(self):
        log.info('-----------------prepare_data: Load data-----------------')
        # log.info(f"Diff Flow Datamodule => {sys._getframe().f_code.co_name}()")
        # load scalers
        log.info(f'data path: {self.data_root}')

    def setup(self, stage: Optional[str] = None) -> None:
        log.info('-----------------setup, construct dataset-----------------')
        if self.is_inited_with_std:  # get the data with std or not
            self.input_scaler = jl.load(os.path.join(self.data_root, 'input_scaler.sav'))
            self.output_scaler = jl.load(os.path.join(self.data_root, 'output_scaler.sav'))

        # load pipeline for conversion from motion features to BV H.
        self.data_pipe = jl.load(
            os.path.join(self.data_root, 'data_pipe_' + str(self.framerate) + 'fps.sav'))
        if stage in (None, "fit") or stage is None:
            log.info(f'-----------------setup stage: {stage}')
            self.train_input = \
                np.load(os.path.join(self.data_root, 'train_input_' + str(self.framerate) + 'fps.npz'))[
                    'clips'].astype(np.float32)  # Train Audio input [8428,120,27] [B,S,F]
            log.info(f'train_input size: {np.shape(self.train_input)}')
            self.train_output = \
                np.load(os.path.join(self.data_root, 'train_output_' + str(self.framerate) + 'fps.npz'))[
                    'clips'].astype(np.float32)  # Train Gesture output [8428,120,45] [B,S,F]

            # log.info(f'self train_output mean: \n {mean_data}, \n {mean_data.shape}')
            self.val_input = \
                np.load(os.path.join(self.data_root, 'val_input_' + str(self.framerate) + 'fps.npz'))[
                    'clips'].astype(np.float32)  # Value Audio input[264,120,27] [B,S,F]
            self.val_output = \
                np.load(os.path.join(self.data_root, 'val_output_' + str(self.framerate) + 'fps.npz'))[
                    'clips'].astype(np.float32)  # Value Gesture output[264,120,45] [B,S,F]
            # log.info(np.shape(train_input) + np.shape(train_output) + np.shape(val_input) + np.shape(val_output))
            # Create pytorch data sets
            if not self.train_dataset and not self.validation_dataset:
                log.info('Construct train dataset and validataion dataset...')
                self.train_dataset = MotionDataset(cond_data=self.train_input, joint_data=self.train_output,
                                                   framerate=self.framerate,
                                                   seqlen=self.seqlen, n_lookahead=self.n_lookahead,
                                                   dropout=self.dropout, is_waveform=self.is_waveform)
                self.validation_dataset = MotionDataset(cond_data=self.val_input, joint_data=self.val_output,
                                                        framerate=self.framerate,
                                                        seqlen=self.seqlen,
                                                        n_lookahead=self.n_lookahead,
                                                        dropout=self.dropout,
                                                        is_waveform=self.is_waveform)
        if stage in (None, "test")  or stage is None:
            log.info(f'-----------------setup stage: {stage}')
            # test data for network tuning. It contains the same data as val_input, but sliced into longer 20-sec
            # exerpts
            # make sure the test data is at least one batch size

            self.test_input = \
                np.load(
                    os.path.join(self.data_root,
                                 'test_input_' + self.test_style + '_' + str(self.framerate) + 'fps.npz'))[
                    'clips'].astype(np.float32)  # [3,400,27]

            if not self.test_dataset:
                self.n_test = self.test_input.shape[0]  # 3
                # initialise test output with zeros (mean pose)
                # self.n_x_channels = self.output_scaler.mean_.shape[0]
                self.n_x_channels = self.gesture_features  # For no standardize
                self.n_test = self.test_input.shape[0]  # 3
                n_tiles = 1 + self.batch_size // self.n_test  # 11
                # [66 ,400,27]
                # test_output = np.tile(self.test_output.copy(), (n_tiles, 1, 1))
                if self.is_waveform:
                    # test_input = np.tile(self.test_input.copy(), (n_tiles, 1))
                    test_output = np.zeros(
                        (self.batch_size,
                         self.test_input.shape[1] // self.audio_sample_rate * self.framerate,
                         self.n_x_channels)).astype(
                        np.float32)  # [66,400,45]
                else:  # MFCC
                    # test_input = np.tile(self.test_input.copy(), (n_tiles, 1, 1))
                    test_output = np.zeros(
                        (self.test_input.shape[0], self.test_input.shape[1], self.n_x_channels)).astype(
                        np.float32)  # [66,400,45]
                    # test_input = np.tile(self.test_input.copy(), (n_tiles, 1, 1))  # [66 ,400,27]
                    # test_output = np.tile(test_output.copy(), (n_tiles, 1, 1))
                self.test_dataset = TestDataset(self.test_input, is_waveform=self.is_waveform)

    def n_channels(self):
        return self.n_x_channels, self.n_cond_channels

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            # persistent_workers=True,
            shuffle=True,
            # drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            # persistent_workers=True,
            # drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_input.shape[0],
            # batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            # persistent_workers=True,
            drop_last=False
        )

    def save_animation(self, motion_data, filename, paramValue):
        print('-----save animation-------------')
        # print(f'motion_data shape: {motion_data.shape}')
        if self.is_inited_with_std:
            anim_clips = inv_standardize(motion_data[:, :, :], self.output_scaler)
        else:
            anim_clips = motion_data[:, :, :]
        # if self.is_inited_with_std:
        #     anim_clips = inv_standardize(motion_data[:self.n_test, :, :], self.output_scaler)
        # else:
        #     anim_clips = motion_data[:self.n_test, :, :]
        log.info("saving generated gestures...")
        if self.save_np:
            np.savez(filename + "_without_smooth.npz", clips=anim_clips)
        # self.write_full_bvh(anim_clips, filename + "_no_sm", paramValue)
        self.write_full_bvh(anim_clips, filename, paramValue)

        if self.is_smoothing:
            log.info("Smoothing generated gestures...")
            smooth_anim_clips = savgol_filter(anim_clips, window_length=self.window_length,
                                              polyorder=self.polyorder, mode='nearest', axis=1)
            if self.save_np:
                np.savez(filename + "_with_smooth.npz", clips=smooth_anim_clips)
            # self.write_bvh(smooth_anim_clips, filename+self.is_with_transcript, paramValue)
            self.write_full_bvh(smooth_anim_clips, filename + "_sm",
                                paramValue)
        # else:

        # self.write_bvh(anim_clips, filename+self.is_with_transcript,paramValue)

    def write_bvh(self, anim_clips, filename, paramValue):
        print('inverse_transform...')
        inv_data = self.data_pipe.inverse_transform(anim_clips)
        writer = BVHWriter()
        for i in range(0, anim_clips.shape[0]):
            if i < 20:
                filename_ = f'{filename}_{paramValue}_{str(i)}.bvh'
                print('writing:' + filename_)
                with open(filename_, 'w') as f:
                    writer.write(inv_data[i], f, framerate=self.framerate)

    def write_full_bvh(self, anim_clip, filename, paramValue):
        log.info('Saving full BVH')
        shape = anim_clip.shape
        combine_anim_clip = anim_clip.reshape((shape[0] * shape[1], shape[2]))
        # log.info('Start smoothing......')
        # combine_anim_clip = savgol_filter(combine_anim_clip, window_length=31,
        #                                   polyorder=4, mode='nearest', axis=0)
        combine_anim_clip = np.expand_dims(combine_anim_clip, axis=0)
        inv_data = self.data_pipe.inverse_transform(combine_anim_clip)

        writer = BVHWriter()
        filename_ = f'{filename}_{self.test_style}{paramValue}.bvh'
        with open(filename_, 'w') as f:
            writer.write(inv_data[0], f, framerate=20)
