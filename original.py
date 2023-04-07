from pysistant import helpers
import os
import hashlib
import re
import librosa
import torch
from tqdm import tqdm
from dataset_creation import load_from_file
from models_usage import MelGanTool
source_dir = './speech_commands_v0.01'
source_files = [item for item in helpers.find_files(source_dir, '.wav')]
import warnings
warnings.simplefilter("ignore")
import torchaudio


MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M


def get_class_name(file_path):
    return file_path.split('/')[-2]


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


def run_notebook():
    def which_set(filename, validation_percentage, testing_percentage):
        """Determines which data partition the file should belong to.

        We want to keep files in the same training, validation, or testing sets even
        if new ones are added over time. This makes it less likely that testing
        samples will accidentally be reused in training when long runs are restarted
        for example. To keep this stability, a hash of the filename is taken and used
        to determine which set it should belong to. This determination only depends on
        the name and the set proportions, so it won't change as other files are added.

        It's also useful to associate particular files as related (for example words
        spoken by the same person), so anything after '_nohash_' in a filename is
        ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
        'bobby_nohash_1.wav' are always in the same set, for example.

        Args:
          filename: File path of the data sample.
          validation_percentage: How much of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.

        Returns:
          String, one of 'training', 'validation', or 'testing'.
        """
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


    print('split')
    train_files = []
    val_files = []
    test_files = []
    class_to_id = {}
    dict_duration = {}
    set_sr = set()
    max_shape = 0
    id = 0
    for file_path in source_files:
        class_name = get_class_name(file_path)

        if class_name not in class_to_id.keys():
            class_to_id[class_name] = id
            id += 1

        samples, sample_rate = librosa.load(file_path, sr=None)
        if len(samples.shape) > 1:
            max_shape = samples.shape

        duration = str(len(samples) / sample_rate)
        if duration in dict_duration.keys():
            dict_duration[duration] += 1
        else:
            dict_duration[duration] = 1

        set_sr.add(sample_rate)

        part_type = which_set(file_path, 10, 10)
        if part_type == 'training':
            train_files.append(file_path)
        elif part_type == 'validation':
            val_files.append(file_path)
        elif part_type == 'testing':
            test_files.append(file_path)

    CLASS_TO_ID = class_to_id

    dataset = dict()
    dataset["train"] = load_from_file(train_files, CLASS_TO_ID)
    dataset["validation"] = load_from_file(val_files, CLASS_TO_ID)
    dataset["test"] = load_from_file(test_files, CLASS_TO_ID)

    import random
    import torch
    import torchaudio
    from torch.utils.data import DataLoader, Dataset, random_split


    class SoundDS(Dataset):

        def __init__(self, source_files=[], class_to_id={}):
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
            return self.source_files[idx]

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

        def append_file(self, spectrogram):
            self.augmentated_data.append(spectrogram)

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

    print('sound')

    train_dataset = SoundDS(source_files=dataset["train"], class_to_id=class_to_id)
    valid_dataset = SoundDS(source_files=dataset["validation"], class_to_id=class_to_id)
    test_dataset = SoundDS(source_files=dataset["test"], class_to_id=class_to_id)

    batch_size = 64

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    import torchvision.models as models

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = models.resnet18().to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print('before_train')

    def train(dataloader, model, loss_fn, optimizer):

        size = len(dataloader.dataset)
        model.train()
        exceptions = list()
        batch = -1
        try:

            for batch, (X, y) in enumerate(tqdm(dataloader)):
                X, y = X.to(device), y.to(device)
                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        except Exception as e:
            print(e)
            exceptions.append(batch)
        print(exceptions)


    def validation(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Val Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    from sklearn.metrics import classification_report


    def test(dataloader, model):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        correct = 0
        y_true = list()
        y_pred = list()
        with torch.no_grad():
            for X, y in dataloader:
                y_true.append(y)
                X, y = X.to(device), y.to(device)
                pred = model(X)
                y_pred.append(pred)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        correct /= size

        # print(classification_report(torch.cat(y_true), torch.cat(y_pred)))
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}\n")
        return (y_true, y_pred)



    # с 1 по 3 эпохи
    epochs = 6
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        validation(val_dataloader, model, loss_fn)


    # model.eval()
    y_true, y_pred = test(test_dataloader, model)

    y_true_tensor = torch.cat((torch.cat(y_true[:1+1]),torch.cat(y_true[1+1:])))
    y_pred_tensor = torch.cat((torch.cat(y_pred[:1+1]),torch.cat(y_pred[1+1:])))
    print(classification_report(y_true_tensor.tolist(), y_pred_tensor.argmax(1).tolist()))
    print("Done!")


run_notebook()