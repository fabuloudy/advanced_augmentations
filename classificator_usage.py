import torchvision.models as models
import torch
from metrics import count_metrics

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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}\n")
    return y_true, y_pred, correct*100

def run_resnet(train_dataloader, val_dataloader, test_dataloader, matrix_filename, weights_file_name = None):
    remember_test_result = {'y_true': [], 'y_pred': []}
    current_score = 0.0
    remember_epoch = 0
    model = models.resnet18().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 30
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        validation(val_dataloader, model, loss_fn)
        y_true, y_pred, correct = test(test_dataloader, model)
        if correct > current_score:
            current_score = correct
            remember_test_result = {'y_true': y_true, 'y_pred': y_pred}
            remember_epoch = epoch + 1

    count_metrics(remember_test_result['y_true'],
                  remember_test_result['y_pred'],
                  matrix_filename, current_score, remember_epoch)
    print("Done!")

