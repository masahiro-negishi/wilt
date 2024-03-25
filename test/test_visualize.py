import os
import sys

import pytest
from torch_geometric.datasets import TUDataset  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from path import DATA_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore
from visualize import visualize_graph, visualize_WLLT  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, depth, withweight",
    [("MUTAG", 2, True), ("MUTAG", 3, False), ("NCI1", 1, True), ("NCI1", 2, False)],
)
def test_visualize_WLLT(tmpdir, dataset_name: str, depth: int, withweight: bool):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    visualize_WLLT(tree, os.path.join(tmpdir, "test_visualize_WLLT.png"), withweight)
    os.remove(os.path.join(tmpdir, "test_visualize_WLLT.png"))


@pytest.mark.parametrize(
    "dataset_name, node_dict",
    [
        ("MUTAG", None),
        ("MUTAG", {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}),
        ("NCI1", None),
    ],
)
def test_visualize_graph(tmpdir, dataset_name: str, node_dict: dict):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    visualize_graph(
        data[0], os.path.join(tmpdir, "test_visualize_graph.png"), node_dict
    )
    os.remove(os.path.join(tmpdir, "test_visualize_graph.png"))
