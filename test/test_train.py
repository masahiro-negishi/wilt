import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from typing import Optional

from torch_geometric.datasets import TUDataset  # type: ignore

from path import DATA_DIR  # type: ignore
from train import TripletSampler, cross_validation, train  # type: ignore
from tree import WeisfeilerLemanLabelingTree  # type: ignore


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
    "dataset_name, depth, normalize, loss_name, batch_size, n_epochs, lr, save_interval, seed, clip_param_threshold, hyperparameter",
    [
        ("MUTAG", 2, True, "triplet", 10, 2, 0.01, 5, 0, None, 1),
        ("MUTAG", 3, False, "nce", 20, 1, 0.001, 10, 42, 1e-3, 1),
    ],
)
def test_train(
    tmpdir,
    dataset_name: str,
    depth: int,
    normalize: bool,
    loss_name: str,
    batch_size: int,
    n_epochs: int,
    lr: float,
    save_interval: int,
    seed: int,
    clip_param_threshold: Optional[float],
    hyperparameter: float,
):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    train_data = data[: len(data) // 2]
    eval_data = data[len(data) // 2 :]
    tree = WeisfeilerLemanLabelingTree(
        data, depth, clip_param_threshold is None, normalize
    )
    if loss_name == "triplet":
        train(
            train_data,
            eval_data,
            tree,
            loss_name,
            batch_size,
            n_epochs,
            lr,
            str(tmpdir),
            save_interval,
            seed,
            clip_param_threshold,
            margin=hyperparameter,
        )
    else:
        train(
            train_data,
            eval_data,
            tree,
            loss_name,
            batch_size,
            n_epochs,
            lr,
            str(tmpdir),
            save_interval,
            seed,
            clip_param_threshold,
            temperature=hyperparameter,
        )
    assert os.path.exists(os.path.join(str(tmpdir), "rslt.json"))
    assert os.path.exists(os.path.join(str(tmpdir), "loss.png"))
    assert os.path.exists(os.path.join(str(tmpdir), "loss_log.png"))
    assert os.path.exists(os.path.join(str(tmpdir), "model_final.pt"))
    for i in range(0, n_epochs // save_interval):
        assert os.path.exists(
            os.path.join(str(tmpdir), f"model_{(i+1) * save_interval}.pt")
        )
    os.remove(os.path.join(str(tmpdir), "rslt.json"))
    os.remove(os.path.join(str(tmpdir), "loss.png"))
    os.remove(os.path.join(str(tmpdir), "loss_log.png"))
    os.remove(os.path.join(str(tmpdir), "model_final.pt"))
    for i in range(0, n_epochs // save_interval):
        os.remove(os.path.join(str(tmpdir), f"model_{(i+1) * save_interval}.pt"))


@pytest.mark.parametrize(
    "dataset_name, k_fold, depth, normalize, loss_name, batch_size, n_epochs, lr, save_interval, seed, clip_param_threshold, hyperparameter",
    [
        ("MUTAG", 5, 2, True, "triplet", 10, 2, 0.01, 5, 0, None, 1),
        ("MUTAG", 10, 3, False, "nce", 20, 1, 0.001, 10, 42, 1e-3, 1),
    ],
)
def test_cross_validation(
    tmpdir,
    dataset_name: str,
    k_fold: int,
    depth: int,
    normalize: bool,
    loss_name: str,
    batch_size: int,
    n_epochs: int,
    lr: float,
    save_interval: int,
    seed: int,
    clip_param_threshold: Optional[float],
    hyperparameter: float,
):
    if loss_name == "triplet":
        cross_validation(
            dataset_name,
            k_fold,
            depth,
            normalize,
            loss_name,
            batch_size,
            n_epochs,
            lr,
            str(tmpdir),
            save_interval,
            seed,
            clip_param_threshold,
            margin=hyperparameter,
        )
    else:
        cross_validation(
            dataset_name,
            k_fold,
            depth,
            normalize,
            loss_name,
            batch_size,
            n_epochs,
            lr,
            str(tmpdir),
            save_interval,
            seed,
            clip_param_threshold,
            temperature=hyperparameter,
        )
    for i in range(k_fold):
        assert os.path.exists(os.path.join(str(tmpdir), f"fold_{i}", "rslt.json"))
        assert os.path.exists(os.path.join(str(tmpdir), f"fold_{i}", "loss.png"))
        assert os.path.exists(os.path.join(str(tmpdir), f"fold_{i}", "loss_log.png"))
        assert os.path.exists(os.path.join(str(tmpdir), f"fold_{i}", "model_final.pt"))
        for i in range(0, n_epochs // save_interval):
            assert os.path.exists(
                os.path.join(
                    str(tmpdir), f"fold_{i}", f"model_{(i+1) * save_interval}.pt"
                )
            )
        os.remove(os.path.join(str(tmpdir), f"fold_{i}", "rslt.json"))
        os.remove(os.path.join(str(tmpdir), f"fold_{i}", "loss.png"))
        os.remove(os.path.join(str(tmpdir), f"fold_{i}", "loss_log.png"))
        os.remove(os.path.join(str(tmpdir), f"fold_{i}", "model_final.pt"))
        for i in range(0, n_epochs // save_interval):
            os.remove(
                os.path.join(
                    str(tmpdir), f"fold_{i}", f"model_{(i+1) * save_interval}.pt"
                )
            )
    assert os.path.exists(os.path.join(str(tmpdir), "info.json"))
    os.remove(os.path.join(str(tmpdir), "info.json"))
