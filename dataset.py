import os
from pathlib import Path
from typing import Iterable, Any, Tuple

import numpy as np
from numpy import ndarray
from scipy import io
from scipy.sparse.linalg import eigsh
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def squash(array):
    # Find unique values in the array
    unique_values = np.unique(array)
    # Create a mapping from original values to new values
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    # Map the original array to new values
    new_array = np.vectorize(value_to_index.get)(array)
    return new_array


def voxelize(spatial, response):
    # Squash position [4,4,4,5,6,6] -> [0,0,0,1,2,2]
    xs = squash(spatial[:, 0])
    ys = squash(spatial[:, 1])
    zs = squash(spatial[:, 2])

    # Calculate grid dimensions
    grid_width = len(np.unique(xs))
    grid_height = len(np.unique(ys))
    grid_depth = len(np.unique(zs))

    # Create an empty 3D grid
    grid = np.zeros((grid_width, grid_height, grid_depth))

    # Map points to the grid
    for neuron_idx in range(spatial.shape[0]):
        # Extract X, Y, Z coordinates from your point cloud data
        x, y, z = xs[neuron_idx], ys[neuron_idx], zs[neuron_idx]

        # Assign intensity value to a voxel value to indicate strength of a signal
        grid[x, y, z] = response[neuron_idx]
    return grid


def encode_labels(labels_mat, raw_labels):
    labels_names = np.array([a[0] for a in labels_mat["class_names"].flatten()])
    labels_map = np.take(labels_names, labels_mat["class_assignment"]).ravel()
    encoded_labels = np.array([np.where(labels_map == label)[0][0] for label in raw_labels])
    return encoded_labels


def decode_labels(labels_mat, encoded_labels):
    labels_names = np.array([a[0] for a in labels_mat["class_names"].flatten()])
    labels_map = np.take(labels_names, labels_mat["class_assignment"]).ravel()
    decoded_labels = labels_map[encoded_labels]
    return decoded_labels


def load_mat(path: Path) -> tuple[ndarray, ndarray]:
    save_path_spatial = path.parent / (path.stem + "_spatial.npy")
    save_path_responses = path.parent / (path.stem + "_responses.npy")
    save_path_labels = path.parent / (path.stem + "_labels.npy")
    if not save_path_spatial.exists() or not save_path_labels.exists() or not save_path_responses.exists():
        dat = io.loadmat(str(path))

        responses: ndarray = dat['stim'][0]['resp'][0]  # stim x neurons
        spont = dat['stim'][0]['spont'][0]  # timepts x neurons
        istim = (dat['stim'][0]['istim'][0]).astype(np.int32)  # stim ids
        istim -= 1  # get out of MATLAB convention
        istim = istim[:, 0]
        nimg = istim.max()  # these are blank stims (exclude them)
        responses = responses[istim < nimg, :]
        labels = istim[istim < nimg].astype(np.uint16)

        # subtract spont (32D)
        mu = spont.mean(axis=0)
        sd = spont.std(axis=0) + 1e-6
        responses = (responses - mu) / sd
        spont = (spont - mu) / sd
        sv, u = eigsh(spont.T @ spont, k=32)
        responses = responses - (responses @ u) @ u.T

        # mean center each neuron
        responses -= responses.mean(axis=0)
        spatial = dat["med"]

        assert responses.shape[0] == labels.shape[0], "Wrong shapes, incorrect data op, needs to be examined"
        assert responses.shape[1] == spatial.shape[0], "Wrong shapes, incorrect data op, needs to be examined"

        # Assuming point cloud has shape (R, N, 4) where:
        # R is the number of responses,
        # N is the number of neurons,
        # and 4 columns are x, y, z, and intensity
        spatial_point_cloud = np.zeros((responses.shape[0], spatial.shape[0], 3), dtype=np.uint16)
        responses = np.zeros((responses.shape[0], spatial.shape[0], 1), dtype=np.float32)

        pbar: Any = tqdm(responses)
        for r_idx, r in enumerate(pbar):
            for i in range(spatial.shape[0]):
                spatial_point_cloud[r_idx, i] = [spatial[i, 0], spatial[i, 1], spatial[i, 2]]
                responses[r_idx, i] = r[i]

        np.save(str(save_path_spatial), spatial_point_cloud)
        np.save(str(save_path_responses), responses)
        np.save(str(save_path_labels), labels)

    spatial_point_cloud = np.load(str(save_path_spatial))
    responses = np.load(str(save_path_responses))
    numpy_point_cloud = np.concatenate((spatial_point_cloud, responses), axis=2)
    labels = np.load(str(save_path_labels))
    return numpy_point_cloud, labels


class MouseBrainPointCloudDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_cloud = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            point_cloud = self.transform(point_cloud)

        return point_cloud, label
