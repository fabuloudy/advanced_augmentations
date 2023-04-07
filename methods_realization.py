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
import torch
import numpy as np
from scipy.signal import butter,filtfilt
import numpy as np
# Filter requirements.
import torchaudio.transforms as transform

from audio_visualization import play_audio, display_audio, display_spectrogram, get_mel_spectrogram, plot_waveform
from augmentation_utils import batch


# InaudibleVoiceCommands
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def unaudible_voice_command(samples, fs = 16000,
                                     cutoff = 8000,
                                     nyq = 1,
                                     order = 2,
                                     n1 = 0,
                                     n2 = 0,
                                     return_wav_format = True):

    # Low-Pass Filtering
    nf = nyq * fs  # Nyquist Frequency
    low_pass_filtered_waveform = butter_lowpass_filter(samples, cutoff, nf, order)

    # Upsampling
    resample_rate = 192000
    resampler = transform.Resample(16000, resample_rate)
    resampled_waveform = resampler(torch.Tensor(low_pass_filtered_waveform.copy()))
    if n1 != 0:
        resampled_waveform *= n1
    else:
        resampled_waveform = 1 / max(abs(resampled_waveform)) * resampled_waveform

    # Ultrasound Modulation
    carrier_freq = 30000
    dt = 1 / resample_rate
    t = np.arange(0, resample_rate*dt, dt)
    a = list(map(lambda x: math.cos(2 * math.pi * carrier_freq * x), t))
    c = np.multiply(resampled_waveform[0], a)

    # Carrier Wave Addition
    b = list(map(lambda x: math.cos(2 * math.pi * carrier_freq * x), t))
    ultrasound = c + torch.Tensor(b)
    if n2 != 0:
        ultrasound *= n2
    else:
        ultrasound =  1/max(abs(ultrasound)) * ultrasound
    # conversion for excluding problems with sound playing
    if return_wav_format:
        samples = [ultra * 0x8000 if ultra < 0  else ultra * 0x7fff for ultra in ultrasound]
        return torch.Tensor(samples).unsqueeze(0)
    return ultrasound.unsqueeze(0)

# TimeDomainInversion

def time_domain_inversion(samples):
    tmp = samples.squeeze(0)
    new_sample = []
    for i in batch(tmp, 10):
        new_sample += reversed(i)
    return torch.Tensor(new_sample).unsqueeze(0)

# AdditionHighFrequencies
def high_frequency_addition(samples, n_fft=1024, n_additional_stft = 32, raising_frequency = 32):
    stft_coefficients = stft(samples, n_fft=n_fft, return_complex=True).squeeze(0)
    stft_coefficients = torch.cat([stft_coefficients,
                                   stft_coefficients[:n_additional_stft] + raising_frequency])
    new_n_fft = (n_additional_stft + int(n_fft / 2)) * 2
    return istft(stft_coefficients.unsqueeze(0), n_fft=new_n_fft)


# PhaseGeneration
def random_phase_generation(samples):
    stft_coefficients = stft(samples, n_fft=1024, return_complex=True).squeeze(0)

    for i in range(len(stft_coefficients)):
        tmp = stft_coefficients[i]
        for j in range(len(tmp)):
            # count magnitude
            # magnitude = abs(sqrt(real^2 + imag^2))
            y = torch.Tensor([math.fabs(math.sqrt(
                math.pow(float(tmp[j].real), 2) + math.pow(float(tmp[j].imag), 2)))]).type(torch.float)[0]
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



# скрещивание
header_len = 44
def crossover(x1, x2):
    ba1 = x1
    ba2 = x2
    step = 2
    # if bps == 8:
    #    step = 1
    for i in range(header_len, len(x1), step):
        if np.random.random() < 0.5:
            ba2[i] = ba1[i]
    return ba2


# мутация
mutation_p = 0.0005
data_max = 32767
data_min = -32768
def mutation(x, eps_limit=256):

    #if pbs == 8:
    #    step = 1
    for i in range(header_len, len(x)):
        #if np.random.random() < 0.05:
        # ba[i] = max(0, min(255, np.random.choice(list(range(ba[i]-4, ba[i]+4)))))
        #elif np.random.random() < 0.10:
        #ba[i] = max(0, min(255, ba[i] + np.random.choice([-1, 1])))
        if np.random.random() < mutation_p:
            new_int_x = min(data_max, max(data_min, x[i] + np.random.choice(range(-eps_limit, eps_limit))))
            x[i] = new_int_x

    return x

# hidden voice commands - туда обратно MFCC

from librosa.feature.inverse import mfcc_to_audio
from torchaudio.transforms import MFCC

def hidden_voice_commands(samples):
    transformator = MFCC()
    mfcc = transformator(samples)
    return mfcc_to_audio(mfcc.numpy())

"""
data, samples_rate = torchaudio.load('C:\\Users\\yulch\\PycharmProjects\\testAugmentations\\speech_commands_v0.01\\bird\\0a7c2a8d_nohash_0.wav')
res = random_phase_generation(data)
torchaudio.save('tmp9.wav',
                            res,
                            16000)
"""