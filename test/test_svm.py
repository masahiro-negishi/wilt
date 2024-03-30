import os
import sys

import torch
from torch_geometric.datasets import TUDataset  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from path import DATA_DIR  # type: ignore
from svm import svm  # type: ignore


def test_svm(fixture_prepare_distances):
    dataset_name, _, path = fixture_prepare_distances
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    distances = torch.load(path)
    indices = list(range(len(data)))
    svm(
        data,
        indices[: len(data) // 2],
        indices[len(data) // 2 : len(data) * 3 // 4],
        indices[len(data) * 3 // 4 :],
        distances,
        1,
    )
