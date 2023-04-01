# GeneticAttack https://github.com/nesl/adversarial_audio
# GeneticAttackModification https://github.com/rtaori/Black-Box-Audio


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torchaudio
from scipy import signal
import plotly.graph_objects as go
from audio_visualization import plot_waveform

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

from audio_visualization import play_audio, display_audio, display_spectrogram, get_mel_spectrogram, plot_waveform
from augmentation_utils import AugmentationMethod

# InaudibleVoiceCommands делаем сами # то же самое что и высокие частоты


def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


import numpy as np
from scipy.signal import butter,filtfilt
# Filter requirements.
import torchaudio.transforms as transform

data, samples_rate = torchaudio.load('C:\\Users\\yulch\\PycharmProjects\\testAugmentations\\speech_commands_v0.01\\bird\\0a7c2a8d_nohash_0.wav')
T = 1.0         # Sample Period
fs = 16000       # sample rate, Hz
cutoff = 8000      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples
print('here')
# Filter the data, and plot both the original and filtered signals.

import torch
#plot_waveform(data, 16000)
y = butter_lowpass_filter(data, cutoff, fs, order)
#plot_waveform(torch.Tensor(y.copy()), 16000)
resample_rate = 192000
resampler = transform.Resample(16000, resample_rate)
resampled_waveform = resampler(torch.Tensor(y.copy()))
resampled_waveform = 1 / max(abs(resampled_waveform)) * resampled_waveform

carrier_freq = 30000

dt = 1 / resample_rate

import numpy as np
t = np.arange(0, resample_rate*dt, dt)
print(t)
print(math.cos(2 * math.pi * carrier_freq * 1.250000e-04))

a = list(map(lambda x: math.cos(2 * math.pi * carrier_freq * x), t))
b = list(map(lambda x: math.cos(2 * math.pi * carrier_freq * x), t))
c = np.multiply(resampled_waveform[0], a)
ultrasound = c + torch.Tensor(b)
ultrasound =  1/max(abs(ultrasound)) * ultrasound


sample = [ultra * 0x8000 if ultra < 0  else ultra * 0x7fff for ultra in ultrasound]
plot_waveform(torch.Tensor(sample).unsqueeze(0),resample_rate)
torchaudio.save(filepath = "tmp5.wav", src = torch.Tensor(sample).unsqueeze(0), sample_rate=resample_rate)
print(ultrasound.unsqueeze(0))

# TimeDomainInversion

def time_domain_inversion(samples):
    tmp = samples.squeeze(0)
    new_sample = []
    for i in batch(tmp, 10):
        new_sample += reversed(i)
    return torch.Tensor(new_sample).unsqueeze(0)

# AdditionHighFrequencies
def high_frequency_addition(samples):
    stft_coefficients = stft(samples, n_fft=1024, return_complex=True).squeeze(0)
    print(stft_coefficients[30:].shape)
    print(stft_coefficients.shape)
    stft_coefficients = torch.cat((stft_coefficients, stft_coefficients+20))
    return istft(stft_coefficients.unsqueeze(0), n_fft=2050)


# PhaseGeneration
def random_phase_generation(samples):
    stft_coefficients = stft(samples, n_fft=1024, return_complex=True).squeeze(0)

    minmax_raw_values = []
    for raw in stft_coefficients:
        real_max = raw[0].real
        real_min = raw[0].real
        for amplitude in raw:
            if amplitude.real > real_max:
                real_max = amplitude.real
            if amplitude.real < real_min:
                real_min = amplitude.real
        minmax_raw_values.append({'real_max': real_max,
                                  'real_min': real_min})

    for i in range(len(stft_coefficients)):
        tmp = stft_coefficients[i]
        for j in range(len(tmp)):
            y = torch.Tensor([math.sqrt(
                math.pow(float(tmp[j].real), 2) + math.pow(float(tmp[j].imag), 2))]).type(torch.float)[0]
            local_real_min, local_real_max = -y, y
            amplitude_real = random.uniform(local_real_min, local_real_max)
            sign = 1 if random.random() < 0.5 else -1
            try:
                amplitude_imag = math.sqrt(math.pow(y, 2) - math.pow(amplitude_real, 2))
            except Exception:
                raise Exception
            complex_amplitude = torch.complex(torch.Tensor([amplitude_real]).type(torch.float),
                                              sign * torch.Tensor([amplitude_imag]).type(torch.float))

            stft_coefficients[i][j] = complex_amplitude.unsqueeze(0)

    return istft(stft_coefficients.unsqueeze(0), n_fft=1024)

#Splice OUT https://arxiv.org/pdf/2110.00046.pdf

def time_mask(spec, T=5, num_masks=1, replace_with_zero=False, splice_out=False):
    cloned = spec.clone()
    cloned2 = spec.clone()
    len_spectro = cloned.shape[2]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        if t >= len_spectro:
            return cloned
        t_zero = random.randrange(0, len_spectro - t)
        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if splice_out:
            a = cloned2[0][:, :t_zero - 1]
            b = cloned2[0][:, mask_end:]
            new_spec = torch.cat((a, b), -1)
            print(new_spec.shape)
            new_dim_spec = torch.cat([new_spec.unsqueeze(0), new_spec.unsqueeze(0), new_spec.unsqueeze(0)], 0)
            return new_dim_spec
        elif replace_with_zero:
            cloned[0][:, t_zero:mask_end] = 0
        else:
            cloned[0][:, t_zero:mask_end] = cloned.mean()
    print(cloned.shape)
    return cloned


