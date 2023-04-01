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


class Classificator:

    def __init__(self, model: Module, loss_fn, optimizer, device: str):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train(self, dataloader):
        size = len(dataloader.dataset)
        self.model.train()
        exceptions = list()
        batch = 0
        try:
            for batch, (x, y) in enumerate(tqdm(dataloader, ncols=80, ascii='True', desc='Train')):
                x, y = x.to(self.device), y.to(self.device)
                # Compute prediction error
                pred = self.model(x)
                loss_batch = self.loss_fn(pred, y)

                # Backpropagation
                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()


                if batch % 100 == 0:
                    loss_batch, current = loss_batch.item(), batch * len(x)
                    print(f"loss: {loss_batch:>7f}  [{current:>5d}/{size:>5d}]")
        except Exception:
            exceptions.append(batch)
        print(f'Problem batches that occurred during model training: {exceptions}.')

    def validation(self, dataloader):

        self.model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for batch, (x, y) in enumerate(tqdm(dataloader, ncols=80, ascii='True', desc='Validation')):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss /= num_batches
        correct /= size

        print(f"Val Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def test(self, dataloader):

        self.model.eval()

        correct = 0
        y_true = list()
        y_pred = list()

        with torch.no_grad():
            for batch, (x, y) in enumerate(tqdm(dataloader, ncols=80, ascii='True', desc='Test')):
                y_true.append(y)

                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                y_pred.append(pred)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        size = len(dataloader.dataset)
        correct /= size

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}\n")

        return y_true, y_pred



def run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename, weights_file_name = None):
    model = models.resnet18().to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    resnet_classificator = Classificator(model, loss_fn, optimizer, DEVICE)

    epochs = 3

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        resnet_classificator.train(train_dataloader)
        resnet_classificator.validation(val_dataloader)
        y_true, y_pred = resnet_classificator.test(test_dataloader)
    resnet_classificator.model.eval()
    count_metrics(y_true, y_pred, matrix_filename)




    print("Done!")
