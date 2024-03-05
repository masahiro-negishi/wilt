import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, depth, n_nodes",
    [
        ("MUTAG", 4, 7 + 33 + 174 + 572 + 1197),
    ],
)
def test_WeisfeilerLemanLabelingTree(dataset_name, depth, n_nodes):
    wwllt = WeisfeilerLemanLabelingTree(dataset_name, depth)
    assert wwllt.n_nodes == n_nodes
