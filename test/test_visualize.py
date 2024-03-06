import os
import sys

import pytest
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.datasets import TUDataset  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from path import DATA_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore
from visualize import tSNE, visualize_graph, visualize_WLLT  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, depth, withweight",
    [("MUTAG", 2, True), ("MUTAG", 3, False)],
)
def test_visualize_WLLT(tmpdir, dataset_name, depth, withweight):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    wwllt = WeisfeilerLemanLabelingTree(data, depth)
    visualize_WLLT(wwllt, os.path.join(tmpdir, "test_visualize_WLLT.png"), withweight)
    os.remove(os.path.join(tmpdir, "test_visualize_WLLT.png"))


@pytest.mark.parametrize(
    "dataset_name, node_dict",
    [
        ("MUTAG", None),
        ("MUTAG", {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}),
    ],
)
def test_visualize_graph(tmpdir, dataset_name, node_dict):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    visualize_graph(
        data[0], os.path.join(tmpdir, "test_visualize_graph.png"), node_dict
    )
    os.remove(os.path.join(tmpdir, "test_visualize_graph.png"))


@pytest.mark.parametrize(
    "dataset_name, depth",
    [("MUTAG", 1), ("MUTAG", 2)],
)
def test_tSNE(tmpdir, dataset_name, depth):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    wwllt = WeisfeilerLemanLabelingTree(data, depth)
    tSNE(wwllt, data, os.path.join(tmpdir, "test_tSNE.png"))
    os.remove(os.path.join(tmpdir, "test_tSNE.png"))
