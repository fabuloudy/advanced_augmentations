from pysistant import helpers
from models_usage import MelGanTool, DiffWaveTool
from dataset_creation import split_train_val_test, load_from_file, load_from_file_augment
from dataset_creation import SoundDS
import torch
from augmentation_utils import get_batch
from classificator_usage import run_resnet
from tqdm import tqdm
from methods_realization import mutation, crossover

CLASS_TO_ID = {}
BATCH_SIZE = 64
from methods_realization import time_domain_inversion, \
    high_frequency_addition, random_phase_generation, butter_lowpass_filter


def prepare_dataset(dataset_tmp):
    dataset = {}
    dataset["train"] = SoundDS(source_files=dataset_tmp["train"], class_to_id=CLASS_TO_ID)
    dataset["validation"] = SoundDS(source_files=dataset_tmp["validation"], class_to_id=CLASS_TO_ID)
    dataset["test"] = SoundDS(source_files=dataset_tmp["test"], class_to_id=CLASS_TO_ID)

    train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset["validation"], batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=BATCH_SIZE, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


def time_domain_inversion_augment_data(dataset):
    augmented_data = []
    for audio in tqdm(dataset):
        audio_new = time_domain_inversion(audio["samples"])
        augmented_data.append({"samples": audio_new.cpu(),
                               "class_id": audio["class_id"]})
    return augmented_data

def high_frequency_addition_augment_data(dataset):
    augmented_data = []
    for audio in tqdm(dataset):
        audio_new = high_frequency_addition(audio["samples"],
                                            n_fft=1024,
                                            n_additional_stft=32,
                                            raising_frequency=32)
        augmented_data.append({"samples": audio_new.cpu(),
                               "class_id": audio["class_id"]})
    return augmented_data

def random_phase_generation_augment_data(dataset):
    augmented_data = []
    for audio in tqdm(dataset):
        audio_new = random_phase_generation(audio["samples"])
        augmented_data.append({"samples": audio_new.cpu(),
                               "class_id": audio["class_id"]})
    return augmented_data

def time_domain_inversion_test(dataset_tmp, matrix_filename):
    augmented_audio = time_domain_inversion_augment_data(dataset_tmp["train"])
    dataset_tmp["train"] = dataset_tmp["train"] + augmented_audio
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)

def high_frequency_addition_test(dataset_tmp, matrix_filename):
    augmented_audio = time_domain_inversion_augment_data(dataset_tmp["train"])
    dataset_tmp["train"] = dataset_tmp["train"] + augmented_audio
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)


def download_random_phase_gen_audio():
    files = [item for item in helpers.find_files('./random_phase_gen', '.wav')]
    class_to_id = {}
    for i in range(31):
        class_to_id[str(i)] = i
    return files, class_to_id

def random_phase_generation_test(dataset_tmp, matrix_filename):
    augmented_audio, class_to_id = download_random_phase_gen_audio()
    dataset_tmp["train"] += load_from_file_augment(augmented_audio, class_to_id)
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)

import torchaudio
def random_phase_generation_audio(dataset_tmp, batch_number):
    batches = [train_batch for train_batch in get_batch(dataset_tmp["train"], 10000)]
    augmented_audio = random_phase_generation_augment_data(batches[batch_number])
    for i in tqdm(range(len(augmented_audio))):
        name = f"./random_phase_gen/batch_{batch_number}_class_{str(augmented_audio[i]['class_id'])}_number_{str(i)}.wav"
        if torch.is_tensor(augmented_audio[i]["samples"]):
            torchaudio.save(name,
                            augmented_audio[i]["samples"],
                            16000)
        else:
            torchaudio.save(name,
                            torch.from_numpy(augmented_audio[i]["samples"]),
                            16000)

def mutation_augment_data(dataset):
    augmented_data = []
    for audio in tqdm(dataset):
        audio_new = mutation(audio["samples"])
        augmented_data.append({"samples": audio_new.cpu(),
                               "class_id": audio["class_id"]})
    return augmented_data

def download_mutation_audio():
    files = [item for item in helpers.find_files('./mutation', '.wav')]
    class_to_id = {}
    for i in range(31):
        class_to_id[str(i)] = i
    return files, class_to_id
def mutation_test(dataset_tmp, matrix_filename):
    augmented_audio, class_to_id = download_mutation_audio()
    dataset_tmp["train"] += load_from_file_augment(augmented_audio, class_to_id)
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)

import numpy as np
def crossover_augment_data(dataset):

    augmented_data = []
    np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, False)
    for _ in tqdm(dataset):
        audio_new = mutation(audio["samples"])
        augmented_data.append({"samples": audio_new.cpu(),
                               "class_id": audio["class_id"]})
    return augmented_data


def mutation_audio(dataset_tmp, batch_number):
    batches = [train_batch for train_batch in get_batch(dataset_tmp["train"], 10000)]
    augmented_audio = mutation_augment_data(batches[batch_number])
    for i in tqdm(range(len(augmented_audio))):
        name = f"./mutation/batch_{batch_number}_class_{str(augmented_audio[i]['class_id'])}_number_{str(i)}.wav"
        if torch.is_tensor(augmented_audio[i]["samples"]):
            torchaudio.save(name,
                            augmented_audio[i]["samples"],
                            16000)
        else:
            torchaudio.save(name,
                            torch.from_numpy(augmented_audio[i]["samples"]),
                            16000)

def crossover_test(dataset_tmp, matrix_filename):
    augmented_audio = crossover_augment_data(dataset_tmp["train"])
    dataset_tmp["train"] = dataset_tmp["train"] + augmented_audio
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)


def main():

    files = [item for item in helpers.find_files('./speech_commands_v0.01', '.wav')]
    dataset_tmp = split_train_val_test(files, r'_nohash_.*$')
    class_to_id = dataset_tmp["class_to_id"]

    global CLASS_TO_ID
    CLASS_TO_ID = class_to_id

    dataset = dict()
    dataset["train"] = load_from_file(dataset_tmp["train"], class_to_id)
    dataset["validation"] = load_from_file(dataset_tmp["validation"], class_to_id)
    dataset["test"] = load_from_file(dataset_tmp["test"], class_to_id)
    #time_domain_inversion_test(dataset, 'tdi_conf_matrix.png')
    #high_frequency_addition_test(dataset, 'hfa_conf_matrix.png')
    #random_phase_generation_test(dataset, 'rfg_conf_matrix.png')
    #random_phase_generation_test(dataset, 'rpg_conf_matrix.png')
    mutation_test(dataset, 'mutation_conf_matrix.png')



main()