import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from typing import Optional

from torch import stft, istft
import math
import random
import torch
from scipy.signal import butter,filtfilt
import numpy as np
import torchaudio.transforms as transform
from librosa.feature.inverse import mfcc_to_audio
from torchaudio import transforms

from augmentation_utils import batch

# InaudibleVoiceCommands
def butter_lowpass_filter(data: torch.Tensor, cutoff: int, nyq: int, order: int):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def inaudible_voice_command(samples: torch.Tensor,
                            sample_rate: int = 16000,
                            cutoff: int = 8000,
                            nyq: int = 1,
                            order: int = 2,
                            n1: int = 0,
                            n2: int = 0,
                            resample_rate = 192000,
                            carrier_freq = 30000,
                            return_wav_format = True) -> torch.Tensor:
    # Low-Pass Filtering
    nf = nyq * sample_rate  # Nyquist Frequency
    low_pass_filtered_waveform = butter_lowpass_filter(samples, cutoff, nf, order)

    # Upsampling
    resample_rate = resample_rate
    resampler = transform.Resample(sample_rate, resample_rate)
    resampled_waveform = resampler(torch.Tensor(low_pass_filtered_waveform.copy()))
    if n1 != 0:
        resampled_waveform *= n1
    else:
        resampled_waveform = 1 / max(abs(resampled_waveform)) * resampled_waveform

    # Ultrasound Modulation
    carrier_freq = carrier_freq
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
def time_domain_inversion(samples: torch.Tensor, window_of_sampling: int = 10) -> torch.Tensor:
    tmp = samples.squeeze(0)
    new_sample = []
    for i in batch(tmp, window_of_sampling):
        new_sample += reversed(i)
    return torch.Tensor(new_sample).unsqueeze(0)

# AdditionHighFrequencies
def high_frequency_addition(samples: torch.Tensor,
                            n_fft: int = 1024,
                            n_additional_stft: int = 32,
                            raising_frequency: int = 32) -> torch.Tensor:
    stft_coefficients = stft(samples, n_fft=n_fft, return_complex=True).squeeze(0)
    stft_coefficients = torch.cat([stft_coefficients,
                                   stft_coefficients[:n_additional_stft] + raising_frequency])
    new_n_fft = (n_additional_stft + int(n_fft / 2)) * 2
    return istft(stft_coefficients.unsqueeze(0), n_fft=new_n_fft)

# PhaseGeneration
def random_phase_generation(samples: torch.Tensor, n_fft: int = 1024) -> torch.Tensor:
    stft_coefficients = stft(samples, n_fft=n_fft, return_complex=True).squeeze(0)
    for i in range(len(stft_coefficients)):
        tmp = stft_coefficients[i]
        for j in range(len(tmp)):
            # count magnitude
            # magnitude = abs(sqrt(real^2 + imag^2))
            y = torch.Tensor([math.fabs(math.sqrt(
                math.pow(float(tmp[j].real), 2) + math.pow(float(tmp[j].imag), 2)))]).type(torch.float)[0]
            local_real_min, local_real_max = -y, y
            amplitude_imag = random.uniform(local_real_min, local_real_max)
            sign = 1 if random.random() < 0.5 else -1
            try:
                amplitude_real = math.sqrt(math.pow(y, 2) - math.pow(amplitude_imag, 2))
            except Exception:
                raise Exception
            complex_amplitude = torch.complex(torch.Tensor([amplitude_real]).type(torch.float),
                                              sign * torch.Tensor([amplitude_imag]).type(torch.float))
            stft_coefficients[i][j] = complex_amplitude.unsqueeze(0)
    return istft(stft_coefficients.unsqueeze(0), n_fft=n_fft)

# скрещивание

def crossover(x1: torch.Tensor,
              x2: torch.Tensor,
              header_length: int = 44,
              skip_step: int = 2,) -> torch.Tensor:
    x1 = x1.squeeze(0)
    x2 = x2.squeeze(0)
    if len(x1) > len(x2):
        x1, x2 = x2, x1
    for i in range(header_length, len(x2), skip_step):
        if np.random.random() < 0.5:
            x2[i] = x1[i]
    return x2.unsqueeze(0)

# мутация

def mutation(x: torch.Tensor,
             header_length: int = 44,
             mutation_p: float = 0.0005,
             data_max: int = 32767,
             data_min: int = -32768) -> torch.Tensor:

    x = x.squeeze(0)
    max_x = abs(max(x))
    min_x = abs(min(x))
    eps_limit = max_x if max_x > min_x else min_x
    for i in range(header_length, len(x)):
        if np.random.random() < mutation_p:
            new_int_x = min(data_max, max(data_min, x[i] + random.uniform(-eps_limit, eps_limit)))
            x[i] = new_int_x
    return x.unsqueeze(0)

# hidden voice commands - туда обратно MFCC
def hidden_voice_commands(samples: torch.Tensor,
                          sample_rate = 16000,
                          in_n_ftt: int = 400,
                          n_mfcc: int = 40,
                          in_win_length: int = 400,
                          in_hop_length: int = 200,
                          out_n_ftt: int = 400,
                          out_win_lenght: int = 400,
                          out_hop_lenght: int = 100) -> torch.Tensor:

    transformer = transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": in_n_ftt,
                   "win_length": in_win_length,
                   "hop_length": in_hop_length,
                   "n_mels": n_mfcc}
    )
    mfcc_audio = transformer(samples).numpy()
    audio = mfcc_to_audio(mfcc_audio,
                          n_fft = out_n_ftt,
                          hop_length = out_hop_lenght,
                          win_length = out_win_lenght)
    return torch.from_numpy(audio)


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
