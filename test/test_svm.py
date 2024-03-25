import os
import sys

import pytest
from torch_geometric.datasets import TUDataset  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from path import DATA_DIR  # type: ignore
from svm import svm, svm_cross_validation  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, depth, gamma",
    [("MUTAG", 1, 1.0), ("MUTAG", 2, 0.1), ("NCI1", 1, 0.01), ("NCI1", 2, 10.0)],
)
def test_svm(dataset_name, depth, gamma):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    svm(tree, data[: len(data) // 2], data[len(data) // 2 :], gamma)


@pytest.mark.parametrize(
    "dataset_name, depth, gamma",
    [("MUTAG", 2, 10.0), ("MUTAG", 3, 0.01), ("NCI1", 2, 0.1), ("NCI1", 3, 1.0)],
)
def test_svm_cross_validation(dataset_name, depth, gamma):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    svm_cross_validation(tree, data, gamma=gamma)
