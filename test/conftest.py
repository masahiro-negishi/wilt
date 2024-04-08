import os
import sys

import pytest
import torch
from torch_geometric.datasets import TUDataset  # type: ignore

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from path import DATA_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.fixture(scope="session", params=[("MUTAG", 1), ("MUTAG", 2), ("NCI1", 1)])
def fixture_prepare_distances(tmpdir_factory, request):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=request.param[0])
    tree = WeisfeilerLemanLabelingTree(data, request.param[1])
    dists = torch.stack(
        [tree.calc_distribution_on_tree(graph) for graph in data],
        dim=0,
    )
    subtree_weights = torch.vmap(tree.calc_subtree_weight)(dists)
    distances = torch.tensor(
        [
            [
                tree.calc_distance_between_subtree_weights(
                    subtree_weights[i], subtree_weights[j]
                ).item()
                for j in range(len(data))
            ]
            for i in range(len(data))
        ]
    )
    f = tmpdir_factory.mktemp("distances").join(
        f"distances_{request.param[0]}_{request.param[1]}.pt"
    )
    torch.save(distances, str(f))
    return request.param[0], request.param[1], str(f)
