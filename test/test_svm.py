import os
import sys

import pytest
from torch_geometric.datasets import TUDataset  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from path import DATA_DIR  # type: ignore
from svm import svm, svm_cross_validation  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, depth",
    [("MUTAG", 1), ("MUTAG", 2)],
)
def test_svm(tmpdir, dataset_name, depth):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    svm(tree, data[: len(data) // 2], data[len(data) // 2 :])


@pytest.mark.parametrize(
    "dataset_name, depth",
    [("MUTAG", 2), ("MUTAG", 3)],
)
def test_svm_cross_validation(tmpdir, dataset_name, depth):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    svm_cross_validation(tree, data)
