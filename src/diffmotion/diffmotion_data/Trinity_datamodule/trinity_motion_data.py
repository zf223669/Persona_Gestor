######################################################3
##          Without gesture condition              ###
######################################################

import numpy as np
import torch
from torch.utils.data import Dataset
from src import utils

log = utils.get_pylogger(__name__)


class MotionDataset(Dataset):
    def __init__(self, cond_data, joint_data, dropout, framerate, seqlen, n_lookahead, is_waveform=False):
        self.joint_data = joint_data
        self.control_data = cond_data
        self.is_waveform = is_waveform

    def __len__(self):
        return self.joint_data.shape[0]

    def __getitem__(self, idx):
        if not self.is_waveform:
            sample = {'joint_data': self.joint_data[idx, :, :], 'cond': self.control_data[idx, :, :]}
        else:
            sample = {'joint_data': self.joint_data[idx, :, :], 'cond': self.control_data[idx, :]}
        return sample


class TestDataset(Dataset):
    def __init__(self, cond_data, is_waveform=False):
        self.control_data = cond_data
        self.is_waveform = is_waveform

    def __len__(self):
        return self.control_data.shape[0]

    def __getitem__(self, idx):
        if not self.is_waveform:
            sample = {'cond': self.control_data[idx, :, :]}
        else:
            sample = {'cond': self.control_data[idx, :]}
        return sample
