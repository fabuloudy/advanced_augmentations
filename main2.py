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

    global CLASS_TO_ID
    CLASS_TO_ID = class_to_id

    dataset = dict()
    dataset["train"] = load_from_file(dataset_tmp["train"], class_to_id)
    dataset["validation"] = load_from_file(dataset_tmp["validation"], class_to_id)
    dataset["test"] = load_from_file(dataset_tmp["test"], class_to_id)
    #time_domain_inversion_test(dataset)

main()