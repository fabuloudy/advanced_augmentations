from pysistant import helpers
from models_usage import MelGanTool, DiffWaveTool
from dataset_creation import split_train_val_test, load_from_file, load_from_file_augment
from dataset_creation import SoundDS
import torch
from augmentation_utils import get_batch
from classificator_usage import run_resnet

CLASS_TO_ID = {}
BATCH_SIZE = 64
"""
y_true_tensor = torch.cat((torch.cat(y_true[:1+1]),torch.cat(y_true[1+1:])))
y_pred_tensor = torch.cat((torch.cat(y_pred[:1+1]),torch.cat(y_pred[1+1:])))
print(classification_report(y_true_tensor.tolist(), y_pred_tensor.argmax(1).tolist()))
"""


def prepare_dataset(dataset_tmp):
    dataset = {}
    dataset["train"] = SoundDS(source_files=dataset_tmp["train"], class_to_id=CLASS_TO_ID)
    dataset["validation"] = SoundDS(source_files=dataset_tmp["validation"], class_to_id=CLASS_TO_ID)
    dataset["test"] = SoundDS(source_files=dataset_tmp["test"], class_to_id=CLASS_TO_ID)

    train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset["validation"], batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=BATCH_SIZE, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


def original_test(dataset_tmp, matrix_filename):
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)


def mel_gan_test(dataset_tmp, matrix_filename):

    melgan_tool = MelGanTool()
    augmented_audio = melgan_tool.augment_data(dataset_tmp["train"])
    dataset_tmp["train"] = dataset_tmp["train"] + augmented_audio
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)

def download_diffwave_audio():
    files = [item for item in helpers.find_files('./diffwave_audio', '.wav')]
    class_to_id = {}
    for i in range(31):
        class_to_id[str(i)] = i
    return files, class_to_id

def diffwave_test(dataset_tmp, matrix_filename, weights_filename):
    augmented_audio, class_to_id = download_diffwave_audio()
    dataset_tmp["train"] += load_from_file_augment(augmented_audio, class_to_id)
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)

import soundfile as sf
import torchaudio

def diffwave_audio(dataset_tmp, batch_number):
    diffwave_tool = DiffWaveTool()
    batches = [train_batch for train_batch in get_batch(dataset_tmp["train"], 10000)]
    augmented_audio = diffwave_tool.augment_data(batches[batch_number])
    for i in range(len(augmented_audio)):
        name = f"./diffwave_audio/batch_{batch_number}_class_{str(augmented_audio[i]['class_id'])}_number_{str(i)}.wav"
        if torch.is_tensor(augmented_audio[i]["samples"]):
            torchaudio.save(name,
                            augmented_audio[i]["samples"],
                            16000)
        else:
            torchaudio.save(name,
                            torch.from_numpy(augmented_audio[i]["samples"]),
                            16000)

def time_domain_inversion_test(dataset_tmp):
    melgan_tool = MelGanTool()
    augmented_audio = melgan_tool.augment_data(dataset_tmp["train"])
    dataset_tmp["train"] = dataset_tmp["train"] + augmented_audio
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(dataset_tmp)
    run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename)

def main():
    files = [item for item in helpers.find_files('./speech_commands_v0.01', '.wav')]
    dataset_tmp = split_train_val_test(files, r'_nohash_.*$')
    class_to_id = dataset_tmp["class_to_id"]
    dataset = dict()
    dataset["train"] = load_from_file(dataset_tmp["train"], class_to_id)
    dataset["validation"] = load_from_file(dataset_tmp["validation"], class_to_id)
    dataset["test"] = load_from_file(dataset_tmp["test"], class_to_id)
    #original_test(dataset, 'original_conf_matrix.png')
    #mel_gan_test(dataset, 'melgan_conf_matrix.png')
    #diffwave_test(dataset, 'diffwave_conf_matrix.png', 'diffwave_weights_state.pth')

main()