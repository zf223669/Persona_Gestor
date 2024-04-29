from scipy.signal import savgol_filter
import numpy as np
from src.data.pymo.writers import *


def inv_standardize(self, data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled


def save_animation(motion_data, filename, n_test:int,  is_smoothing: bool=False, paramValue: str =""):
    print('-----save animation-------------')
    # print(f'motion_data shape: {motion_data.shape}')
    # control_data = control_data.cpu().numpy()
    # motion_data = motion_data.cpu().numpy()
    anim_clips = inv_standardize(motion_data[:n_test, :, :], output_scaler)
    if is_smoothing:
        smooth_anim_clips = savgol_filter(anim_clips, window_length=30,
                                          polyorder=4, mode='nearest', axis=1)
        # self.showJointData(anim_clips, smooth_anim_clips, filename)
        # print(f'anim_clips shape: {anim_clips.shape}')
        np.savez(filename + ".npz", clips=smooth_anim_clips)
        write_bvh(smooth_anim_clips, filename, paramValue)
    else:
        np.savez(filename + ".npz", clips=anim_clips)
        write_bvh(anim_clips, filename)


def write_bvh(self, anim_clips, filename, paramValue):
    print('inverse_transform...')
    inv_data = self.data_pipe.inverse_transform(anim_clips)
    writer = BVHWriter()
    for i in range(0, anim_clips.shape[0]):
        if i < 20:
            filename_ = f'{filename}{paramValue}_{str(i)}.bvh'
            print('writing:' + filename_)
            with open(filename_, 'w') as f:
                writer.write(inv_data[i], f, framerate=self.framerate)
