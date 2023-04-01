import torch
import torchaudio.transforms as TT

class AugmentationMethod:
    def __init__(self, function):
        pass

    def __call__(self, *args, **kwargs):
        pass

def transform(audio, sr = 22050):
    audio = torch.clamp(audio, -1.0, 1.0)

    mel_args = {
        'sample_rate': sr,
        'win_length': 256 * 4,
        'hop_length': 256,
        'n_fft': 1024,
        'f_min': 20.0,
        'f_max': sr / 2.0,
        'n_mels': 80,
        'power': 1.0,
        'normalized': True,
    }
    mel_spec_transform = TT.MelSpectrogram(**mel_args)

    with torch.no_grad():
        spectrogram = mel_spec_transform(audio)
        spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
        spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
        return spectrogram.cpu()

def batch(iterable, n=400):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_batch(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]