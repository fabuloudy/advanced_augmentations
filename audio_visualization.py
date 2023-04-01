import torch
import torchaudio
from torch import stft, istft
import numpy as np
import math
from torchaudio.transforms import MelSpectrogram, MFCC
import torch
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
import IPython.display as ipd
import matplotlib.pyplot as plt
import torchaudio
import random


def get_mel_spectrogram(audioraw):
    transform = MelSpectrogram(16000, n_mels=80)
    mel_specgram = transform(torch.Tensor(audioraw))
    return mel_specgram

def display_spectrogram(spectrogram):
    plt.figure(figsize=(9,4)) # arbitrary, looks good on my screen.
    plt.imshow(spectrogram)
    plt.show()


def play_audio(waveform):
    return Audio(waveform, rate=16000)

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=True)


def display_audio(audio_array):
      audio_tensor = torch.Tensor(audio_array)
      #ipd.Audio(data=audio_tensor, rate=16000)
      audioraw = audio_tensor
      plot_waveform(audioraw, 16000)