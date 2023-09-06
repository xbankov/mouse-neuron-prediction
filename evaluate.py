import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score


def evaluate(model, dataloader, criterion):
    model.eval()
    losses = []
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            outputs = model(X)
            loss = criterion(outputs, y)
            losses.append(loss.item())
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true = y.cpu().numpy()
            y_preds.extend(y_pred)
            y_trues.extend(y_true)

    accuracy = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, average='macro')

    wandb.log({
        "Val Loss": np.mean(losses),
        "Val Accuracy": accuracy,
        "Val F1 Score": f1,
    })

    return np.mean(losses), accuracy, f1
