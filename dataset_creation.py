import hashlib
import os
import random
from random import shuffle
import re
from typing import Union

import torch
from torch import Tensor
import torchaudio
from torch.utils.data import Dataset as torchDataset



MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1

def which_set(filename: str, validation_percentage: Union[int, float],
              testing_percentage: Union[int, float]) -> str:
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.

    hash_name_hashed = hashlib.sha1(str(hash_name).encode('utf-8')).hexdigest()

    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'

    return result



def get_class_name(file_path):
    return file_path.split('/')[-2]


def get_speaker_id(file_path):

    return re.sub('.wav', '', file_path.split('_nohash_')[1])


def split_train_val_test(source_files_paths: list, file_name_template: str):
    train_files = []
    val_files = []
    test_files = []
    class_to_id = {}
    id = 0
    for file_path in source_files_paths:
        class_name = get_class_name(file_path)

        if class_name not in class_to_id.keys():
            class_to_id[class_name] = id
            id += 1

        part_type = which_set(file_path, 10, 10)
        if part_type == 'training':
            train_files.append(file_path)
        elif part_type == 'validation':
            val_files.append(file_path)
        elif part_type == 'testing':
            test_files.append(file_path)


    return {"train": train_files,
            "validation": val_files,
            "test": test_files,
            "class_to_id": class_to_id}



class SoundDS(torchDataset):

    def __init__(self, source_files: list, class_to_id: dict):
        self.data = source_files
        self.duration = 1000  # длительность
        self.sr = 16000  # частота дисркетизации
        self.class_to_id = class_to_id
        self.n_fft = 1024  # количетсво отсчетов на кадр
        self.hop_length = None  # масштаб времени по оси
        self.n_mels = 64  # duration/window_of_fft
        self.top_db = 80  # пороговое значение, дальше - тишина
        self.sample_rate = 16000

    def __len__(self):
        return len(self.data)

    def get_elem(self, idx):
        return self.data[idx]

    def __getitem__(self, idx):
        if idx < len(self.data):
            # выравниваем по длине звук (убираем, добавляем сэмплы)
            elem = self.data[idx]
            samples = self._pad_trunc(elem["samples"], self.sample_rate)
        else:
            raise Exception(f'Index {idx} is larger than the dataset size')

        # spect has shape [channel, n_mels, time], where channel is mono, stereo etc
        spect = torchaudio.transforms.MelSpectrogram(
            self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )(samples)
        spect = torchaudio.transforms.AmplitudeToDB(top_db=self.top_db)(spect)
        spect = self.change_number_of_channels(spect, 3)
        return spect, elem["class_id"]

    def _pad_trunc(self, samples, sr):
        num_rows, signal_len = samples.shape
        max_len = sr // 1000 * self.duration

        if (signal_len > max_len):
            # Truncate the signal to the given length
            samples = samples[:, :max_len]

        elif (signal_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - signal_len)
            pad_end_len = max_len - signal_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((1, pad_begin_len))
            pad_end = torch.zeros((1, pad_end_len))

            samples = torch.cat((pad_begin, samples, pad_end), 1)

        return samples

    def change_number_of_channels(self, spect, num_channel):
        if (spect.shape[0] == num_channel):
            # Nothing to do
            return spect

        if (num_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            spect = spect[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            spect = torch.cat([spect, spect, spect])

        return spect


def load_from_file(source_files: list, class_to_id):
    data = []
    for file_path in source_files:
        try:
            samples, _ = torchaudio.load(file_path, normalize=True)
        except Exception:
            print(1)
            continue
        data.append({"samples": samples, "class_id": class_to_id[get_class_name(file_path)]})
    return data

def get_class_name_augment(filename):
    return filename.split('class_')[1].split('_number')[0]


def load_from_file_augment(source_files: list, class_to_id):
    data = []
    for file_path in source_files:
        try:
            samples, _ = torchaudio.load(file_path, normalize=True)
        except Exception:
            print(1)
            continue
        data.append({"samples": samples, "class_id": class_to_id[get_class_name_augment(file_path)]})
    return data
