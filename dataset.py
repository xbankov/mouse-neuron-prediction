import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from numpy import ndarray
from scipy import io
from scipy.sparse.linalg import eigsh
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def save_label_mappings(root_dir, label_to_int, int_to_label, image_to_idx):
    with open(root_dir / "label_to_int.json", "w") as fp:
        json.dump(label_to_int, fp)
    with open(root_dir / "int_to_label.json", "w") as fp:
        json.dump(int_to_label, fp)
    with open(root_dir / "image_idx_to_label.json", "w") as fp:
        json.dump(image_to_idx, fp)


def load_label_mappings(root_dir):
    with open(root_dir / "label_to_int.json", "r") as fp:
        label_to_int = json.load(fp, object_hook=lambda d: {k: int(v) for k, v in d.items()})
    with open(root_dir / "int_to_label.json", "r") as fp:
        int_to_label = json.load(fp, object_hook=lambda d: {int(k): v for k, v in d.items()})
    with open(root_dir / "image_idx_to_label.json", "r") as fp:
        image_idx_to_label = json.load(fp, object_hook=lambda d: {int(k): v for k, v in d.items()})
    return label_to_int, int_to_label, image_idx_to_label

    # Labels transformation


def transform_labels(labels_mat, root_dir):
    labels = np.array([a[0] for a in labels_mat["class_names"].flatten()])
    labels_count = len(np.unique(labels))
    labels_int = list(map(int, np.arange(labels_count)))
    label_to_int = dict(zip(labels, labels_int))
    int_to_label = dict(zip(labels_int, labels))

    image_labels = np.take(labels, labels_mat["class_assignment"]).ravel()
    image_idx = list(map(int, np.arange(len(labels_mat["class_assignment"][0]))))
    image_idx_to_label = dict(zip(image_idx, image_labels))

    save_label_mappings(root_dir, label_to_int, int_to_label, image_idx_to_label)
    label_to_int, int_to_label, image_idx_to_label = load_label_mappings(root_dir)

    return image_idx_to_label, label_to_int, int_to_label


def load_mat(path: Path) -> Tuple[ndarray, ndarray]:
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
