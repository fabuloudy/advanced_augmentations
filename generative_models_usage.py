import torchaudio
import sys
import torch
from generative_models.melgan.utils.stft import TacotronSTFT
from diffwave.inference import predict as diffwave_predict
from typing import Optional
from tqdm import tqdm
from sklearn import preprocessing
from augmentation_utils import transform
from torch.nn import Module

class MelGanTool:
    def __init__(self, pretrained_model: Module = None, stft: TacotronSTFT = None):
        if pretrained_model:
            self.vocoder = pretrained_model
            self.stft = stft
        else:
            self.vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
            self.stft = TacotronSTFT(filter_length=1024,
                                     hop_length=256,
                                     win_length=1024,
                                     n_mel_channels=80,
                                     sampling_rate=22050,
                                     mel_fmin=0.0,
                                     mel_fmax=8000.0)
        self.vocoder.eval()

    def augment_audio(self, samples: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available():
            self.vocoder.cuda()
        else:
            raise Exception("Cuda is not available")
        wav = samples
        try:
            mel = self.stft.mel_spectrogram(wav)
        except Exception:
            raise Exception('Problem with input data format')
        mel = mel.cuda()
        audio_new = self.vocoder.inference(mel)
        generated_audio = torch.Tensor(preprocessing.normalize(audio_new.unsqueeze(0).cpu()))
        return generated_audio

class DiffWaveTool:
    def __init__(self, pretrained_model_pt: str = None):
        if pretrained_model_pt:
            self.model_dir = pretrained_model_pt
        else:
            self.model_dir = 'generative_models/diffwave/diffwave-ljspeech-22kHz-1000578.pt'

    def augment_audio(self, samples: torch.Tensor) -> torch.Tensor:
        spectrogram = transform(samples)
        audio_new, sample_rate = diffwave_predict(spectrogram, self.model_dir, fast_sampling=True)
        generated_audio = audio_new.cpu()
        return generated_audio
