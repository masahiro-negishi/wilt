import os
import sys

import pytest
import torch
from torch_geometric.data import Data

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, depth, n_nodes",
    [
        ("MUTAG", 2, 7 + 33 + 174),
        ("MUTAG", 4, 7 + 33 + 174 + 572 + 1197),
    ],
)
def test_WeisfeilerLemanLabelingTree_init(dataset_name: str, depth: int, n_nodes: int):
    wwllt = WeisfeilerLemanLabelingTree(dataset_name, depth)
    assert wwllt.n_nodes == n_nodes


@pytest.mark.parametrize(
    "dataset_name, depth",
    [("MUTAG", 1), ("MUTAG", 3)],
)
def test_WeisfeilerLemanLabelingTree_calc_distributionance(
    dataset_name: str, depth: int
):
    wwllt = WeisfeilerLemanLabelingTree("MUTAG", 4)
    graph1 = [wwllt.data[0], wwllt.data[1], wwllt.data[2]]
    graph2 = [wwllt.data[3], wwllt.data[4], wwllt.data[5]]
    distance = wwllt.calc_distance(graph1, graph2)
    for d in distance:
        assert d >= 0
