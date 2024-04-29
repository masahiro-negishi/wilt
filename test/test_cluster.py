import os
import sys

import matplotlib.pyplot as plt  # type: ignore
import torch
from torch_geometric.datasets import TUDataset  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from cluster import intra_inter_distance, tSNE  # type: ignore
from path import DATA_DIR  # type: ignore


def test_tSNE(fixture_prepare_distances):
    dataset_name, _, path = fixture_prepare_distances
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    fig, ax = plt.subplots()
    indices = list(range(len(data)))
    distances = torch.load(path)
    tSNE(data, ax, indices[: len(data) // 2], distances)
    plt.close(fig)


def test_intra_inter_distance(fixture_prepare_distances):
    dataset_name, _, path = fixture_prepare_distances
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    fig, ax = plt.subplots()
    indices = list(range(len(data)))
    distances = torch.load(path)
    intra_inter_distance(
        data, ax, indices[: len(data) // 2], indices[len(data) // 2 :], distances, 10
    )
    plt.close(fig)
