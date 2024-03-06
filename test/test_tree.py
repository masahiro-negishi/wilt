import os
import sys

import pytest
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.datasets import TUDataset  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from path import DATA_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, depth, n_nodes",
    [
        ("MUTAG", 2, 7 + 33 + 174),
        ("MUTAG", 4, 7 + 33 + 174 + 572 + 1197),
    ],
)
def test_WeisfeilerLemanLabelingTree_init(dataset_name: str, depth: int, n_nodes: int):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    wwllt = WeisfeilerLemanLabelingTree(data, depth)
    assert wwllt.n_nodes == n_nodes


@pytest.mark.parametrize(
    "dataset_name, depth",
    [("MUTAG", 1), ("MUTAG", 3)],
)
def test_WeisfeilerLemanLabelingTree_calc_distributionance(
    dataset_name: str, depth: int
):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    wwllt = WeisfeilerLemanLabelingTree(data, depth)
    graph1 = [data[0], data[1], data[2]]
    graph2 = [data[3], data[4], data[5]]
    distance = wwllt.calc_distance(graph1, graph2)
    for d in distance:
        assert d >= 0
