from pysistant import helpers
import torch
import torchvision.models as models
from torch.nn import Module
from torch.nn.modules import loss
from torch import optim
from tqdm import tqdm
from IPython.display import Audio
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import pickle
import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT
from sklearn import preprocessing
from augmentation_utils import get_batch

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import os.path
from metrics import count_metrics

from dataset_creation import SoundDS, split_train_val_test, load_from_file
from models_usage import MelGanTool

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
CLASS_TO_ID = {}

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    exceptions = list()
    batch = -1
    try:
        for batch, (X, y) in enumerate(dataloader):
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
    return y_true, y_pred




def run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename, weights_file_name = None):
    model = models.resnet18().to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 3
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        validation(val_dataloader, model, loss_fn)

    print("Done!")


    epochs = 6
    # model.eval()
    y_true, y_pred = test(test_dataloader, model)
    count_metrics(y_true, y_pred, matrix_filename)


    print("Done!")
