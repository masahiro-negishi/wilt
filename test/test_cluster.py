import os
import sys

import pytest
from torch_geometric.datasets import TUDataset  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from cluster import tSNE  # type: ignore
from path import DATA_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, depth",
    [("MUTAG", 1), ("MUTAG", 2)],
)
def test_tSNE(tmpdir, dataset_name, depth):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    wwllt = WeisfeilerLemanLabelingTree(data, depth)
    tSNE(wwllt, data, os.path.join(tmpdir, "test_tSNE.png"))
    os.remove(os.path.join(tmpdir, "test_tSNE.png"))
