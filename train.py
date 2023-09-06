import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


def train(model, criterion, optimizer, pbar):
    model.train()
    losses = []
    y_preds = []
    y_trues = []
    for i, (X, y) in enumerate(pbar):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true = y.cpu().numpy()

        y_preds.extend(y_pred)
        y_trues.extend(y_true)
        pbar.set_postfix_str(f"Train loss: {loss.item():0.2f}")

    accuracy = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, average='macro')

    pbar.set_postfix_str(f"Train loss: {np.mean(losses):0.2f} | "
                         f"Accuracy: {accuracy:0.2f} | "
                         f"F1 Score: {f1:0.2f}")

    wandb.log({
        "Train Loss": np.mean(losses),
        "Train Accuracy": accuracy,
        "Train F1 Score": f1,
    })

    return np.mean(losses), accuracy, f1
