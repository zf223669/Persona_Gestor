"""Based on: https://github.com/simonalexanderson/StyleGestures"""
import numpy as np
import librosa
import soundfile as sf
import glob
import os
import sys
from shutil import copyfile
from extract_BEAT_motion_features import extract_joint_angles, extract_hand_pos, extract_style_features
from extract_BEAT_audio_features import extract_melspec, extract_waveform
# import scipy.io.wavfile as wav
from src.utils.pymo.parsers import BVHParser
from src.utils.pymo.data import Joint, MocapData
from src.utils.pymo.preprocessing import *
from src.utils.pymo.writers import *
from sklearn.preprocessing import StandardScaler
import joblib as jl
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def fit_and_standardize(data):
    shape = data.shape
    print(f"fit_and_standarize data.shape: {data.shape}")
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler


def standardize(data, scaler):
    shape = data.shape
    print(f"standarize data.shape: {data.shape}")
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))  # StandardScaler expected <= 2
    scaled = scaler.transform(flat).reshape(shape)
    return scaled


def cut_audio(filename, timespan, destpath, starttime=0.0, endtime=-1.0):
    print(f'Cutting AUDIO {filename} into intervals of {timespan}')
    X, fs = librosa.load(filename, sr=None)
    # fs, Y = wav.read(filename)
    if endtime <= 0:
        endtime = len(X) / fs
    suffix = 0
    while (starttime + timespan) <= endtime:
        out_basename = os.path.splitext(os.path.basename(filename))[0]
        # wav_outfile = out_basename + "_" + str(suffix).zfill(3) + '.wav'
        wav_outfile = os.path.join(destpath, out_basename + "_" + str(suffix).zfill(3) + '.wav')
        start_idx = int(np.round(starttime * fs))
        end_idx = int(np.round((starttime + timespan) * fs)) + 1
        if end_idx >= X.shape[0]:
            return

        sf.write(wav_outfile, X[start_idx:end_idx], samplerate=int(fs), subtype='PCM_24')
        # wav.write(wav_outfile, fs, X[start_idx:end_idx])
        starttime += timespan
        suffix += 1


def cut_bvh(filename, timespan, destpath, starttime=0.0, endtime=-1.0):
    print(f'Cutting BVH {filename} into intervals of {timespan}')

    p = BVHParser()
    bvh_data = p.parse(filename)
    if endtime <= 0:
        endtime = bvh_data.framerate * bvh_data.values.shape[0]

    writer = BVHWriter()
    suffix = 0
    while (starttime + timespan) <= endtime:
        out_basename = os.path.splitext(os.path.basename(filename))[0]
        bvh_outfile = os.path.join(destpath, out_basename + "_" + str(suffix).zfill(3) + '.bvh')
        start_idx = int(np.round(starttime / bvh_data.framerate))
        end_idx = int(np.round((starttime + timespan) / bvh_data.framerate)) + 1
        if end_idx >= bvh_data.values.shape[0]:
            return

        with open(bvh_outfile, 'w') as f:
            writer.write(bvh_data, f, start=start_idx, stop=end_idx)

        starttime += timespan
        suffix += 1


def slice_data(data, window_size, overlap):
    nframes = data.shape[0]
    overlap_frames = (int)(overlap * window_size)

    n_sequences = (nframes - overlap_frames) // (window_size - overlap_frames)
    sliced = np.zeros((n_sequences, window_size, data.shape[1])).astype(np.float32)

    if n_sequences > 0:

        # extract sequences from the data
        for i in range(0, n_sequences):
            frameIdx = (window_size - overlap_frames) * i
            sliced[i, :, :] = data[frameIdx:frameIdx + window_size, :].copy()
    else:
        print("WARNING: data too small for window")

    return sliced


def align(data1, data2):
    """Truncates to the shortest length and concatenates"""

    nframes1 = data1.shape[0]
    nframes2 = data2.shape[0]
    if nframes1 < nframes2:
        return np.concatenate((data1, data2[:nframes1, :]), axis=1)
    else:
        return np.concatenate((data1[:nframes2, :], data2), axis=1)


def import_data(file, motion_path, speech_path, mirror=False, start=0, end=None):
    """Loads a file and concatenate all features to one [time, features] matrix.
     NOTE: All sources will be truncated to the shortest length, i.e. we assume they
     are time synchronized and has the same start time."""

    suffix = ""
    if mirror:
        suffix = "_mirrored"

    motion_data = np.load(os.path.join(motion_path, file + suffix + '.npz'))['clips'].astype(np.float32)
    n_motion_feats = motion_data.shape[1]

    speech_data = np.load(os.path.join(speech_path, file + '.npy')).astype(np.float32)

    control_data = speech_data
    concat_data = align(motion_data, control_data)

    if not end:
        end = concat_data.shape[0]

    return concat_data[start:end, :], n_motion_feats


def import_and_slice(files, motion_path, speech_path, slice_window, slice_overlap, mirror=False, start=0,
                     end=None):
    """Imports all features and slices them to samples with equal lenth time [samples, timesteps, features]."""

    fi = 0
    for file in files:
        print(f'import_and_slice: {file}')

        # slice dataset
        concat_data, n_motion_feats = import_data(file, motion_path, speech_path, False, start, end)
        sliced = slice_data(concat_data, slice_window, slice_overlap)

        if mirror:
            concat_mirr, nmf = import_data(file, motion_path, speech_path, True, start, end)
            sliced_mirr = slice_data(concat_mirr, slice_window, slice_overlap)

            # append to the sliced dataset
            sliced = np.concatenate((sliced, sliced_mirr), axis=0)

        if fi == 0:
            out_data = sliced
        else:
            out_data = np.concatenate((out_data, sliced), axis=0)
        fi = fi + 1

    return out_data[:, :, :n_motion_feats], out_data[:, :, n_motion_feats:]


def slice_data_with_waveform(control_data, fps, audio_sample_rate, window_size, overlap):
    n_sequences = (int)(control_data.shape[0] / audio_sample_rate / (window_size / fps))
    overlap_frames = (int)(overlap * window_size)

    # n_sequences = n_sequences
    sliced_control_data = np.zeros(((int)(n_sequences), window_size // fps * audio_sample_rate)).astype(np.float32)

    if n_sequences > 0:
        # extract sequences from the data
        for i in range(0, n_sequences):
            frameIdx = (window_size - overlap_frames) * i
            sliced_control_data[i, :] = control_data[frameIdx // fps * audio_sample_rate:(
                                                                                                     frameIdx + window_size) // fps * audio_sample_rate].copy()
    else:
        print("WARNING: data too small for window")

    return sliced_control_data


def align_with_waveform(data1, data2, fps: int = 20, audio_sample_rate: int = 16000):
    """Truncates to the shortest length and concatenates"""

    n_time1 = data1.shape[0] // fps
    n_time2 = data2.shape[0] // audio_sample_rate
    if n_time1 < n_time2:
        return data1, data2[:n_time1 * audio_sample_rate]
    else:
        return data1[:n_time2 * fps], data2


def import_data_with_waveform(file, speech_path, fps=20, audio_sample_rate=16000, start=0,
                              end=None):
    speech_data = np.load(os.path.join(speech_path, file + '.npy')).astype(np.float32)  # [T,]
    # control_data = speech_data
    # clip_control_data = control_data

    return speech_data[:]


def import_and_slice_with_waveform(files, speech_path, slice_window, slice_overlap, mirror=False, fps=20,
                                   audio_sample_rate=16000, start=0, end=None):
    out_sliced_control_data = None
    fi = 0
    for file in files:
        print(file)
        control_data = import_data_with_waveform(file=file, speech_path=speech_path,
                                                 fps=fps, audio_sample_rate=audio_sample_rate, start=start,
                                                 end=end)
        # slice dataset
        sliced_control_data = slice_data_with_waveform(control_data, fps,
                                                       audio_sample_rate, slice_window,
                                                       slice_overlap)
        if fi == 0:
            out_sliced_control_data = sliced_control_data
        else:
            out_sliced_control_data = np.concatenate((out_sliced_control_data, sliced_control_data), axis=0)

        fi = fi + 1

    return out_sliced_control_data


# @hydra.main(version_base="1.3", config_path="../../../../configs_diffmotion",
#             config_name="data/BEAT/prepare_BEAT_dataset_MFCC.yaml")
# @hydra.main(version_base="1.3", config_path="../../../../configs_diffmotion",
#             config_name="data/BEAT/prepare_BEAT_dataset.yaml")

#prepare_BEAT_only_audio_dataset
#prepare_ZEGGS_only_audio_dataset
#prepare_Trinity_only_audio_dataset
@hydra.main(version_base="1.3", config_path="../../../../configs_diffmotion",
            config_name="data/BEAT/prepare_BEAT_only_audio_dataset.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # print(OmegaConf.to_yaml(cfg))
    cfg_para = cfg.data.BEAT
    '''
        Converts bvh and wav files into features, slices in equal length intervals and divides the data
        into training, validation and test sets. '''

    # Hardcoded preprocessing params and file structure.
    # Modify these if you want the data in some different format
    train_window_secs = cfg_para.train_window_secs  # 6
    test_window_secs = cfg_para.test_window_secs  # 20
    window_overlap = cfg_para.window_overlap  # 0.5
    fps = cfg_para.fps
    is_standardize = cfg_para.is_standardize
    map_to_exponential = cfg_para.map_to_exponential
    extract_MFCC = cfg_para.extract_MFCC
    data_root = cfg_para.data_dir
    n_mels = cfg_para.n_mels
    # pass
    bvhpath = os.path.join(data_root, 'bvh')
    audiopath = data_root
    print('........bvhpath = ' + bvhpath)
    print('........audiopath = ' + audiopath)
    # held_out = [cfg_para.holdout]
    held_out = cfg_para.holdout
    processed_dir = cfg_para.processed_dir
    audio_sample_rate = cfg_para.audio_sample_rate

    files = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(audiopath):
        for file in sorted(f):
            if '.wav' in file:
                ff = os.path.join(r, file)
                basename = os.path.splitext(os.path.basename(ff))[0]
                files.append(basename)

    # print(ff)

    # processed data will be organized as following
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    motion_feature = 'joint_rot'
    speech_feature = 'audio'

    if not map_to_exponential:
        exponential_state = "NoExp"
    else:
        exponential_state = "WithExp"

    if not extract_MFCC:
        audio_feature = "waveform"
    else:
        audio_feature = f"{n_mels}MFCC"

    if not is_standardize:
        standardize_state = "NotStd"
    else:
        standardize_state = "WithStd"

    path = os.path.join(processed_dir,
                        f'feat_{fps}fps_{train_window_secs}s_{exponential_state}_{audio_feature}_{standardize_state}')
    # motion_path = os.path.join(path, f'{motion_feature}')
    speech_path = os.path.join(path, f'{speech_feature}')

    if not os.path.exists(path):
        os.makedirs(path)

    # speech features
    print('Processing speech features...')
    print(os.path.exists(speech_path))
    if not os.path.exists(speech_path):
        os.makedirs(speech_path)
        extract_waveform(audiopath, files, speech_path, fps, audio_sample_rate)

    print("divide in train, val, dev and test sets.\n Preparing datasets...")

    # train_files = [f for f in files if f not in held_out]

    slice_win_train = train_window_secs * fps
    slice_win_test = test_window_secs * fps
    val_test_split = 20 * test_window_secs * fps  # 10
    input_scaler = jl.load(os.path.join(data_root, 'input_scaler.sav'))
    # input_scaler = os.path.join(data_root, 'input_scaler.sav')
    if is_standardize:
        print("Standardize Processing...........")
        for test_file in held_out:
            temp_test_file_list = [test_file]
            print(f'import_and_slice raw audio: {temp_test_file_list}')
            test_ctrl = import_and_slice_with_waveform(temp_test_file_list, speech_path,
                                                       slice_win_test, 0,
                                                       fps=fps,
                                                       audio_sample_rate=audio_sample_rate, start=0)
            test_ctrl = np.expand_dims(test_ctrl, axis=2).swapaxes(1, 2)
            test_ctrl = standardize(test_ctrl, input_scaler)
            test_ctrl = np.squeeze(test_ctrl, axis=1)
            np.savez(os.path.join(path, f'test_input_{test_file}_{fps}fps.npz'), clips=test_ctrl)
            # test_ctrl = np.squeeze(test_ctrl, axis=1)
    else:
        print("Skip Standardizing......")
    print('Save train/val npz!!!')


if __name__ == "__main__":
    main()
