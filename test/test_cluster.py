import os
import sys

import matplotlib.pyplot as plt  # type: ignore
import pytest
from torch_geometric.datasets import TUDataset  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from cluster import intra_inter_distance, tSNE  # type: ignore
from path import DATA_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, depth",
    [("MUTAG", 1), ("MUTAG", 2)],
)
def test_tSNE(dataset_name, depth):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    fig, ax = plt.subplots()
    indices = list(range(len(data)))
    tSNE(
        tree,
        data,
        ax,
        indices[: len(data) // 2],
        indices[len(data) // 2 :],
    )
    plt.close(fig)


@pytest.mark.parametrize(
    "dataset_name, depth",
    [("MUTAG", 1), ("MUTAG", 2)],
)
def test_intra_inter_distance(dataset_name, depth):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    fig, ax = plt.subplots()
    indices = list(range(len(data)))
    intra_inter_distance(
        tree, data, ax, indices[: len(data) // 2], indices[len(data) // 2 :]
    )
    plt.close(fig)
