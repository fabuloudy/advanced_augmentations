import torchaudio
import sys
import torch
import torch
from complited_solutions.melgan.utils.stft import TacotronSTFT
from diffwave.inference import predict as diffwave_predict
from typing import Optional
from tqdm import tqdm
from sklearn import preprocessing
from augmentation_utils import transform

from augmentation_utils import AugmentationMethod

class MelGanTool:

    def __init__(self):
        self.vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
        self.vocoder.eval()

    def augment_data(self, dataset: list) -> Optional[list]:
        augmented_audio = []
        stft = TacotronSTFT(filter_length=1024,
                            hop_length=256,
                            win_length=1024,
                            n_mel_channels=80,
                            sampling_rate=22050,
                            mel_fmin=0.0,
                            mel_fmax=8000.0)
        if torch.cuda.is_available():
            self.vocoder.cuda()
        else:
            return
        with torch.no_grad():
            for audio in tqdm(dataset):
                wav = audio["samples"]
                try:
                    mel = stft.mel_spectrogram(wav)
                except Exception:
                    print(wav)
                    raise Exception
                mel = mel.cuda()
                audio_new = self.vocoder.inference(mel)
                augmented_audio.append({"samples":
                                            torch.Tensor(preprocessing.normalize(
                                            audio_new.unsqueeze(0).cpu())),
                                        "class_id": audio["class_id"]})

        return augmented_audio


class DiffWaveTool:

    def __init__(self):
        self.model_dir = './complited_solutions/diffwave-ljspeech-22kHz-1000578.pt'

    def augment_data(self, dataset):
        augmented_data = []
        for audio in tqdm(dataset):
            spectrogram = transform(audio["samples"])
            audio_new, sample_rate = diffwave_predict(spectrogram, self.model_dir, fast_sampling=True)
            augmented_data.append({"samples": audio_new.cpu(),
                                   "class_id": audio["class_id"]})
        return augmented_data
