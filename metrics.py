from sklearn.metrics import classification_report
import torch
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt


def count_metrics(y_true, y_pred, matrix_filename):

    y_true_tensor = torch.cat((torch.cat(y_true[:1 + 1]), torch.cat(y_true[1 + 1:])))

    y_pred_tensor = torch.cat((torch.cat(y_pred[:1 + 1]), torch.cat(y_pred[1 + 1:])))
    print(classification_report(y_true_tensor.tolist(), y_pred_tensor.argmax(1).tolist()))

    cm = confusion_matrix(y_true_tensor.tolist(), y_pred_tensor.argmax(1).tolist())
    cmp = ConfusionMatrixDisplay(cm, display_labels=np.arange(30))
    fig, ax = plt.subplots(figsize=(15, 15))
    fig.savefig('foo.png')
    plt.show()
    cmp.plot(ax=ax)

    cmp.figure_.savefig(matrix_filename)