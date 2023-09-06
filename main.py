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

from dataset import load_mat, MouseBrainPointCloudDataset
from evaluate import evaluate
from models import PointCloudNet
from train import train

if __name__ == "__main__":
    wandb.init(project="neuroscience",
               config={
                   "learning_rate": 0.001,
                   "architecture": "PointCloudNet",
                   "dataset": "MouseLand",
                   "epochs": 10,
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
    labels_file: str = "stimuli_class_assignment_confident.mat"
    image_type: str = "natimg2800"
    number_of_stimuli: int = 2800

    path = list(root_dir.glob(f"{image_type}_M*.mat"))[0]
    labels_mat = io.loadmat(str(root_dir / labels_file))

    logger.info("Loading data ...")
    numpy_point_cloud, raw_labels = load_mat(path)

    transform = transforms.Compose([transforms.ToTensor()])

    # Labels transformation
    labels = np.array([a[0] for a in labels_mat["class_names"].flatten()])
    labels_count = len(np.unique(labels))
    labels_int = np.arange(labels_count)
    label_to_int = dict(zip(labels, labels_int))
    int_to_label = dict(zip(labels_int, labels))

    image_labels = np.take(labels, labels_mat["class_assignment"]).ravel()
    image_idx = np.arange(len(labels_mat["class_assignment"][0]))
    image_to_idx = dict(zip(image_idx, image_labels))

    labels_str = np.vectorize(image_to_idx.get)(raw_labels)
    labels_int = np.vectorize(label_to_int.get)(labels_str)

    indices = np.arange(len(labels_int))

    train_val_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
    test_dataset = MouseBrainPointCloudDataset(numpy_point_cloud[test_indices],
                                               labels_int[test_indices],
                                               transform=transform)

    # # K-fold cross-validation on 90% of data
    k = 2
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    logger.info("Training ...")
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
        batch_size = 32
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        logging.debug(f"#batches training dataset: {len(train_dataloader)}")
        logging.debug(f"#batches validation dataset: {len(val_dataloader)}")

        # Create the model
        model = PointCloudNet(num_classes=labels_count, num_neurons=numpy_point_cloud.shape[1])
        wandb.watch(model)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 2
        train_loss = np.nan
        val_loss = np.nan

        for epoch in range(num_epochs):
            pbar: Any = tqdm(train_dataloader)
            losses, accuracy, f1 = train(model, criterion, optimizer, pbar)
            val_losses, val_accuracy, val_f1 = evaluate(model, val_dataloader, criterion)

            pbar.set_description(
                f"Epochs[{epoch}/{num_epochs}] | "
                f"Train F1 Score: {f1:0.2f} | "
                f"Validation F1 Score: {val_f1:0.2f}")

    wandb.finish()
