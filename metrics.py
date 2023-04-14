from sklearn.metrics import classification_report
import torch
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import csv
import pandas as pd

from sklearn.metrics import classification_report
import csv
import pandas as pd



#call the classification_report first and then our new function

def make_report(y_true, y_pred, matrix_filename, epoch):
    labels = list(set(y_true))
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    repdf = pd.DataFrame(report_dict).round(5).transpose()
    repdf.insert(loc=0, column='class', value=labels + ["accuracy", "macro avg", "weighted avg"])
    repdf.to_csv(f"{str(matrix_filename.split('_conf')[0])}_epoch_{epoch}.csv", index=False)

def count_metrics(y_true, y_pred, matrix_filename, correct, epoch):
    y_true_tensor = torch.cat((torch.cat(y_true[:1 + 1]), torch.cat(y_true[1 + 1:])))
    y_pred_tensor = torch.cat((torch.cat(y_pred[:1 + 1]), torch.cat(y_pred[1 + 1:])))
    make_report(y_true_tensor.tolist(),
                y_pred_tensor.argmax(1).tolist(),
                matrix_filename, epoch)

    cm = confusion_matrix(y_true_tensor.tolist(), y_pred_tensor.argmax(1).tolist())
    cmp = ConfusionMatrixDisplay(cm, display_labels=np.arange(30))
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.show()
    cmp.plot(ax=ax)
    cmp.figure_.savefig(f"{str(matrix_filename.split('.')[0])}_epoch_{epoch}.png")