from pysistant import helpers
import torchaudio
from models_usage import MelGanTool, DiffWaveTool
from dataset_creation import split_train_val_test, load_from_file, load_from_file_augment
from dataset_creation import SoundDS
import torch
import numpy as np
from tqdm import tqdm
from augmentation_utils import get_batch
from classificator_usage import run_resnet
from methods_realization import time_domain_inversion, \
    high_frequency_addition, random_phase_generation, butter_lowpass_filter,\
        mutation, crossover, hidden_voice_commands

CLASS_TO_ID = {}
BATCH_SIZE = 64

def hidden_voice_commands_audio(dataset_tmp, batch_number):
    batches = [train_batch for train_batch in get_batch(dataset_tmp["train"], 10000)]
    augmented_audio = hidden_voice_commands_augment_data(batches[batch_number])
    for i in tqdm(range(len(augmented_audio))):
        name = f"./hidden_voice_commands/batch_{batch_number}_class_{str(augmented_audio[i]['class_id'])}_number_{str(i)}.wav"
        if torch.is_tensor(augmented_audio[i]["samples"]):
            torchaudio.save(name,
                            augmented_audio[i]["samples"],
                            16000)
        else:
            torchaudio.save(name,
                            torch.from_numpy(augmented_audio[i]["samples"]),
                            16000)

def prepare_dataset(dataset_tmp):
    dataset = {}
    dataset["train"] = SoundDS(source_files=dataset_tmp["train"], class_to_id=CLASS_TO_ID)
    dataset["validation"] = SoundDS(source_files=dataset_tmp["validation"], class_to_id=CLASS_TO_ID)
    dataset["test"] = SoundDS(source_files=dataset_tmp["test"], class_to_id=CLASS_TO_ID)

    train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset["validation"], batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=BATCH_SIZE, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader

def crossover_augment_data(dataset):
    splitted_data = dict([(i, []) for i in range(30)])
    for audio in dataset:
        splitted_data[audio["class_id"]].append(audio["samples"])
    augmented_data = []
    for class_id, list_audio in splitted_data.items():
        for _ in tqdm(range(len(list_audio))):
            cross_pair = np.random.choice(list_audio, size=2, replace=False)
            audio_new = crossover(cross_pair[0], cross_pair[1])
            augmented_data.append({"samples": audio_new.cpu(),
                               "class_id": class_id})
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


def augment_data(dataset, method_name):
    augmented_data = []
    for audio in tqdm(dataset):
        audio_new = method_name(audio["samples"])
        augmented_data.append({"samples": audio_new.cpu(),
                               "class_id": audio["class_id"]})
    return augmented_data

def augmentation_ram_test(dataset_tmp, matrix_filename, augment_data_function):
    augmented_audio = augment_data_function(dataset_tmp["train"])
    dataset_tmp["train"] = dataset_tmp["train"] + augmented_audio
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)


def augmentation_disk_test(dataset_tmp, matrix_filename, folder_name):
    augmented_audio, class_to_id = download_audio(folder_name)
    dataset_tmp["train"] += load_from_file_augment(augmented_audio, class_to_id)
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)

def download_audio(folder_name):
    files = [item for item in helpers.find_files('./' + folder_name, '.wav')]
    class_to_id = {}
    for i in range(30):
        class_to_id[str(i)] = i
    return files, class_to_id

def augment_and_save_audio(dataset_tmp,
                           batch_number,
                           augment_data_function,
                           folder_name,
                           batch_size = 10000):
    # batch number for long processing
    batches = [train_batch for train_batch in get_batch(dataset_tmp["train"], batch_size)]
    augmented_audio = augment_data_function(batches[batch_number])
    for i in tqdm(range(len(augmented_audio))):
        name = f'./' + folder_name + f"/batch_{batch_number}_class_{str(augmented_audio[i]['class_id'])}_number_{str(i)}.wav"
        if torch.is_tensor(augmented_audio[i]["samples"]):
            torchaudio.save(name,
                            augmented_audio[i]["samples"],
                            16000)
        else:
            torchaudio.save(name,
                            torch.from_numpy(augmented_audio[i]["samples"]),
                            16000)

def random_phase_generation_test(dataset_tmp, matrix_filename, folder_name):
    augmentation_disk_test(dataset_tmp, matrix_filename, folder_name)
def time_domain_inversion_test(dataset_tmp, matrix_filename, augment_data_function):
    augmentation_ram_test(dataset_tmp, matrix_filename, augment_data_function)
def high_frequency_addition_test(dataset_tmp, matrix_filename, augment_data_function):
    augmentation_ram_test(dataset_tmp, matrix_filename, augment_data_function)
def mutation_test(dataset_tmp, matrix_filename, folder_name):
    augmentation_disk_test(dataset_tmp, matrix_filename, folder_name)
def crossover_test(dataset_tmp, matrix_filename, augment_data_function):
    augmentation_ram_test(dataset_tmp, matrix_filename, augment_data_function)
def hidden_voice_commands_test(dataset_tmp, matrix_filename, folder_name):
    augmentation_disk_test(dataset_tmp, matrix_filename, folder_name)

def hidden_voice_commands_test(dataset_tmp, matrix_filename):
    augmented_audio = hidden_voice_commands_augment_data(dataset_tmp["train"])
    dataset_tmp["train"] = dataset_tmp["train"] + augmented_audio
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)

def hidden_voice_commands_augment_data(dataset):
    augmented_data = []
    for audio in tqdm(dataset):
        audio_new = hidden_voice_commands(audio["samples"])
        augmented_data.append({"samples": audio_new.cpu(),
                               "class_id": audio["class_id"]})
    return augmented_data
def main():
    files = [item for item in helpers.find_files('./speech_commands_v0.01', '.wav')]
    dataset_tmp = split_train_val_test(files, r'_nohash_.*$')
    class_to_id = dataset_tmp["class_to_id"]
    dataset = dict()
    dataset["train"] = load_from_file(dataset_tmp["train"], class_to_id)
    dataset["validation"] = load_from_file(dataset_tmp["validation"], class_to_id)
    dataset["test"] = load_from_file(dataset_tmp["test"], class_to_id)
    #time_domain_inversion_test(dataset, 'tdi_conf_matrix.png', time_domain_inversion)
    #high_frequency_addition_test(dataset, 'hfa_conf_matrix.png', high_frequency_addition)
    #random_phase_generation_test(dataset, 'rpg_conf_matrix.png', random_phase_generation)
    #mutation_test(dataset, 'mutation_conf_matrix.png', mutation)
    #crossover_test(dataset, 'crossover_conf_matrix.png', crossover)
    #hidden_voice_commands_test(dataset, 'hvc_conf_matrix.png', hidden_voice_commands)

main()