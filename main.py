import json
from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from scipy import io
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import load_mat, MouseBrainPointCloudDataset, transform_labels
from evaluate import evaluate
from models import PointCloudNet
from train import train


def main(args):
    if args.wandb:
        wandb.init(project="neuroscience",
                   config={
                       "architecture": "PointCloudNet",
                       "dataset": "MouseLand",
                       "epochs": args.epochs,
                   })

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler('logs/train.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Load your 3D data and labels
    root_dir: Path = Path("data")
    # https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/classes/stimuli_class_assignment_confident.mat
    labels_file: str = "stimuli_class_assignment_confident.mat"
    image_type: str = "natimg2800"

    path = list((root_dir / "dataset").glob(f"{image_type}_M*.mat"))[0]
    labels_mat = io.loadmat(str(root_dir / labels_file))

    logger.info("Loading data ...")
    numpy_point_cloud, raw_labels = load_mat(path)
    transform = transforms.Compose([transforms.ToTensor()])

    image_idx_to_label, label_to_int, int_to_label = transform_labels(labels_mat, root_dir)
    labels_count = len(label_to_int.keys())
    labels_str = np.vectorize(image_idx_to_label.get)(raw_labels)
    labels_int = np.vectorize(label_to_int.get)(labels_str)

    indices = np.arange(len(labels_int))

    train_val_indices_path = root_dir / "train_val_indices.npy"
    test_indices_path = root_dir / "test_indices.npy"

    if not train_val_indices_path.exists() or not test_indices_path.exists():
        logger.info("Splitting the dataset and saving the indices into files.")
        train_val_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
        np.save(train_val_indices_path, train_val_indices)
        np.save(test_indices_path, test_indices)

    train_val_indices = np.load(train_val_indices_path)
    test_indices = np.load(test_indices_path)

    test_dataset = MouseBrainPointCloudDataset(numpy_point_cloud[test_indices],
                                               labels_int[test_indices],
                                               transform=transform)

    # # K-fold cross-validation on 90% of data
    k = args.kfold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    logger.info("Training ...")
    logger.info(f"Random accuracy is {1 / labels_count:0.2f}%")
    for fold, (train_index, val_index) in enumerate(kf.split(train_val_indices)):
        logger.info(f"K-Fold: {fold} starting ... ")

        # Train and validate model using train_index and val_index
        train_indices = train_val_indices[train_index]
        val_indices = train_val_indices[val_index]

        # Create a custom dataset
        train_dataset = MouseBrainPointCloudDataset(numpy_point_cloud[train_indices], labels_int[train_indices])
        val_dataset = MouseBrainPointCloudDataset(numpy_point_cloud[val_indices], labels_int[val_indices])

        logging.debug(f"Size of the training dataset: {len(train_dataset)}")
        logging.debug(f"Size of the validation dataset: {len(val_dataset)}")

        # Create a data loader
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        logging.debug(f"#batches training dataset: {len(train_dataloader)}")
        logging.debug(f"#batches validation dataset: {len(val_dataloader)}")

        # Create the model
        model = PointCloudNet(num_classes=labels_count, num_neurons=numpy_point_cloud.shape[1])
        if args.wandb:
            wandb.watch(model)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        t_acc = 0.0
        v_acc = 0.0

        t_loss = 0.0
        v_loss = 0.0
        for epoch in range(args.epochs):
            pbar: Any = tqdm(train_dataloader)

            epoch_str = f"[Epochs: {epoch} / {args.epochs}]"
            loss_str = f"[Avg.Loss: {t_loss: 0.2f} | {v_loss: 0.2f}]"
            acc_str = f"[Acc: {t_acc:0.2f}%|{v_acc:0.2f}%]"

            pbar_prefix = epoch_str + loss_str + acc_str
            t_loss, t_acc, t_f1 = train(model, criterion, optimizer, pbar, pbar_prefix, args)
            v_loss, v_acc, v_f1 = evaluate(model, val_dataloader, criterion, args)

        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for model to train.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--kfold", default=5, type=int, help="Number of splits in KFold CV.")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--wandb", action="store_true", help="Log metrics into WANDB.")
    main(parser.parse_args())
