import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torchaudio
from scipy import signal
import plotly.graph_objects as go
from audio_visualization import plot_waveform
import math
def main():
    pass
"""
    from audio file.
    [voice_signal, voice_samp_freq] = audioread(voice_file);
    [b, a] = butter(10, 2 * 8000 / voice_samp_freq, 'low');
    voice_filter = filter(b, a, voice_signal(:, 1));


    ultra_samp_freq = 192000;
    voice_resamp = resample(voice_filter, ultra_samp_freq, voice_samp_freq);
    voice_resamp = 1 / max(abs(voice_resamp)) * voice_resamp;

    dt = 1 / ultra_samp_freq;
    len = size(voice_resamp, 1);
    t = (0:dt:(len - 1) * dt)';
    carrier_freq = 30000;
    ultrasound = voice_resamp. * cos(2 * pi * carrier_freq * t) + 1 * cos(2 * pi * carrier_freq * t);
    
    ultrasound = 1 / max(abs(ultrasound)) * ultrasound;


    ultrasound_file = '';
    audiowrite(ultrasound_file, ultrasound, ultra_samp_freq);
"""


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



