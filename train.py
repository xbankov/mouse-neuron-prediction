import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score


def train(model, criterion, optimizer, pbar, pbar_prefix, args, device):
    model.train()
    losses = []
    y_preds = []
    y_trues = []
    for _, (X, y) in enumerate(pbar):
        # Move data to the GPU
        X, y = X.to(device), y.to(device)

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
        pbar.set_description(f"{pbar_prefix} | Current loss: {loss.item():0.2f}")

    accuracy = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, average='macro')
    loss = np.mean(losses)
    if args.wandb:
        wandb.log({
            "Train Loss": loss,
            "Train Accuracy": accuracy,
            "Train F1 Score": f1,
        })

    return loss, accuracy, f1
