import os
import sys

import pytest
import torch
from torch_geometric.data import Batch  # type: ignore
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
    tree = WeisfeilerLemanLabelingTree(data, depth)
    assert tree.n_nodes == n_nodes


@pytest.mark.parametrize(
    "dataset_name, depth",
    [("MUTAG", 1), ("MUTAG", 3)],
)
def test_WeisfeilerLemanLabelingTree_calc_distance_between_dists(
    dataset_name: str, depth: int
):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    graph1 = torch.rand(3, tree.n_nodes)
    graph2 = torch.rand(3, tree.n_nodes)
    distance = tree.calc_distance_between_dists(graph1, graph2)
    for d in distance:
        assert d >= 0
    graph1 = torch.rand(tree.n_nodes)
    graph2 = torch.rand(tree.n_nodes)
    distance = tree.calc_distance_between_dists(graph1, graph2)
    assert d >= 0


@pytest.mark.parametrize(
    "dataset_name, depth",
    [("MUTAG", 1), ("MUTAG", 3)],
)
def test_WeisfeilerLemanLabelingTree_calc_distance_between_graphs(
    dataset_name: str, depth: int
):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    graph1 = [data[0], data[1], data[2]]
    graph2 = [data[3], data[4], data[5]]
    distance = tree.calc_distance_between_graphs(graph1, graph2)
    for d in distance:
        assert d >= 0
    graph1 = Batch.from_data_list([data[6], data[7], data[8]])
    graph2 = Batch.from_data_list([data[9], data[10], data[11]])
    distance = tree.calc_distance_between_graphs(graph1, graph2)
    for d in distance:
        assert d >= 0
