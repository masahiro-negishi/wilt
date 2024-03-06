import os
import sys

import pytest
import torch
from torch_geometric.data import Data  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from tree import WeisfeilerLemanLabelingTree  # type: ignore
from visualize import visualize_graph, visualize_WLLT  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, depth, withweight",
    [("MUTAG", 2, True), ("MUTAG", 3, False)],
)
def test_visualize_WLLT(tmpdir, dataset_name, depth, withweight):
    wwllt = WeisfeilerLemanLabelingTree(dataset_name, depth)
    visualize_WLLT(wwllt, os.path.join(tmpdir, "test_visualize_WLLT.png"), withweight)
    os.remove(os.path.join(tmpdir, "test_visualize_WLLT.png"))
