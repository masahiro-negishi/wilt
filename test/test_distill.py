import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from typing import Optional

from torch_geometric.datasets import TUDataset  # type: ignore

from distill import PairSampler, train_gd  # type: ignore
from path import DATA_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, batch_size, train",
    [
        ("MUTAG", 10, True),
        ("MUTAG", 20, False),
        ("Mutagenicity", 10, True),
        ("Mutagenicity", 20, False),
        ("NCI1", 10, True),
        ("NCI1", 20, False),
    ],
)
def test_PairSampler(dataset_name: str, batch_size: int, train: bool):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    sampler = PairSampler(data, batch_size, torch.ones((len(data), len(data))), train)
    for left_indices, right_indices, distances in sampler:
        assert (
            left_indices.shape
            == right_indices.shape
            == distances.shape
            == (batch_size,)
        )
        break
    assert sampler.all_pairs.shape == (len(data) * (len(data) - 1) // 2, 2)


@pytest.mark.parametrize(
    "dataset_name, depth, normalize, seed, loss_name, absolute, l1coeff, batch_size, n_epochs, lr, save_interval, clip_param_threshold",
    [
        ("MUTAG", 2, True, 0, "l1", True, 0.01, 32, 1, 0.01, 1, None),
        ("MUTAG", 3, False, 42, "l2", False, 0, 16, 2, 0.01, 1, 0.0),
    ],
)
def test_train_gd(
    tmpdir,
    dataset_name: str,
    depth: int,
    normalize: bool,
    seed: int,
    loss_name: str,
    absolute: bool,
    l1coeff: float,
    batch_size: int,
    n_epochs: int,
    lr: float,
    save_interval: int,
    clip_param_threshold: Optional[float],
):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(
        data, depth, clip_param_threshold is None, normalize
    )
    train_gd(
        data,
        "tree",
        tree,
        seed,
        str(tmpdir),
        loss_name,
        absolute,
        l1coeff,
        batch_size,
        n_epochs,
        lr,
        save_interval,
        clip_param_threshold,
        torch.ones(len(data), len(data)),
    )
