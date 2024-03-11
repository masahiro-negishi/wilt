import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from torch_geometric.datasets import TUDataset  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore

from path import DATA_DIR  # type: ignore
from train import TripletSampler, train  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, batch_size",
    [
        ("MUTAG", 10),
        ("MUTAG", 20),
    ],
)
def test_TripletSampler(dataset_name: str, batch_size: int):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    sampler = TripletSampler(data, batch_size)
    assert len(sampler) == len(data) // batch_size
    for i, indices in enumerate(sampler):
        if i != len(sampler):
            assert len(indices) == batch_size * 3


@pytest.mark.parametrize(
    "dataset_name, depth, loss_name, batch_size, n_epochs, lr, save_interval, seed, hyperparameter",
    [
        ("MUTAG", 2, "triplet", 10, 2, 0.01, 5, 0, 1),
        ("MUTAG", 3, "nce", 20, 1, 0.001, 10, 42, 1),
    ],
)
def test_train(
    tmpdir,
    dataset_name: str,
    depth: int,
    loss_name: str,
    batch_size: int,
    n_epochs: int,
    lr: float,
    save_interval: int,
    seed: int,
    hyperparameter: float,
):
    if loss_name == "triplet":
        train(
            dataset_name,
            depth,
            loss_name,
            batch_size,
            n_epochs,
            lr,
            str(tmpdir),
            save_interval,
            seed,
            margin=hyperparameter,
        )
    else:
        train(
            dataset_name,
            depth,
            loss_name,
            batch_size,
            n_epochs,
            lr,
            str(tmpdir),
            save_interval,
            seed,
            temperature=hyperparameter,
        )
    assert os.path.exists(os.path.join(str(tmpdir), "info.json"))
    assert os.path.exists(os.path.join(str(tmpdir), "loss.png"))
    assert os.path.exists(os.path.join(str(tmpdir), "loss_log.png"))
    assert os.path.exists(os.path.join(str(tmpdir), "model_final.pt"))
    for i in range(0, n_epochs // save_interval):
        assert os.path.exists(
            os.path.join(str(tmpdir), f"model_{(i+1) * save_interval}.pt")
        )
    os.remove(os.path.join(str(tmpdir), "info.json"))
    os.remove(os.path.join(str(tmpdir), "loss.png"))
    os.remove(os.path.join(str(tmpdir), "loss_log.png"))
    os.remove(os.path.join(str(tmpdir), "model_final.pt"))
    for i in range(0, n_epochs // save_interval):
        os.remove(os.path.join(str(tmpdir), f"model_{(i+1) * save_interval}.pt"))
