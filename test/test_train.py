import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from typing import Optional

from torch_geometric.datasets import TUDataset  # type: ignore

from path import DATA_DIR  # type: ignore
from train import (  # type: ignore
    NPlusTwoSampler,
    TripletSampler,
    cross_validation,
    train,
)
from tree import WeisfeilerLemanLabelingTree  # type: ignore


@pytest.mark.parametrize(
    "dataset_name, batch_size",
    [
        ("MUTAG", 10),
        ("MUTAG", 20),
        ("NCI1", 10),
        ("NCI1", 20),
    ],
)
def test_TripletSampler(dataset_name: str, batch_size: int):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    sampler = TripletSampler(data, batch_size)
    cnt_a = 0
    cnt_p = 0
    cnt_n = 0
    for i, (anchor_indices, positive_indices, negative_indices) in enumerate(sampler):
        cnt_a += len(anchor_indices)
        cnt_p += len(positive_indices)
        cnt_n += len(negative_indices)
        if i != len(sampler) - 1:
            assert (
                len(anchor_indices)
                == len(positive_indices)
                == len(negative_indices)
                == batch_size
            )
    assert cnt_a == cnt_p == cnt_n == len(data)


@pytest.mark.parametrize(
    "dataset_name, batch_size, n_negative",
    [
        ("MUTAG", 10, 5),
        ("MUTAG", 20, 10),
        ("NCI1", 10, 5),
        ("NCI1", 20, 10),
    ],
)
def test_NPlusTwoSampler(dataset_name: str, batch_size: int, n_negative: int):
    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    sampler = NPlusTwoSampler(data, batch_size, n_negative)
    cnt_a = 0
    cnt_p = 0
    cnt_n = 0
    for i, (anchor_indices, positive_indices, negative_indices) in enumerate(sampler):
        cnt_a += len(anchor_indices)
        cnt_p += len(positive_indices)
        cnt_n += len(negative_indices)
        if i != len(sampler) - 1:
            assert (
                len(anchor_indices)
                == len(positive_indices)
                == len(negative_indices)
                == batch_size
            )
        assert len(negative_indices[0]) == n_negative
    assert cnt_a == cnt_p == cnt_n == len(data)


@pytest.mark.parametrize(
    "dataset_name, depth, normalize, loss_name, batch_size, n_epochs, lr, save_interval, seed, clip_param_threshold, hyperparameter",
    [
        ("MUTAG", 2, True, "triplet", 10, 2, 0.01, 5, 0, None, 1),
        ("MUTAG", 3, False, "nce", 20, 1, 0.001, 10, 42, 1e-3, 1),
        ("MUTAG", 2, True, "infonce", 20, 2, 0.01, 5, 0, None, 1),
        ("MUTAG", 2, False, "allpairnce", 188, 2, 0.01, 5, 0, None, 1),
        ("NCI1", 2, True, "triplet", 128, 2, 0.01, 5, 0, 1e-3, 1),
        ("NCI1", 3, False, "nce", 256, 1, 0.001, 10, 42, None, 1),
        ("NCI1", 2, True, "infonce", 256, 2, 0.01, 5, 0, 1e-3, 1),
        ("NCI1", 2, True, "allpairnce", 256, 2, 0.01, 5, 0, 1e-3, 1),
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
            save_interval,
            seed,
            str(tmpdir),
            clip_param_threshold,
            margin=hyperparameter,
        )
    elif loss_name == "nce":
        train(
            train_data,
            eval_data,
            tree,
            loss_name,
            batch_size,
            n_epochs,
            lr,
            save_interval,
            seed,
            str(tmpdir),
            clip_param_threshold,
            temperature=hyperparameter,
        )
    elif loss_name == "infonce":
        train(
            train_data,
            eval_data,
            tree,
            loss_name,
            batch_size,
            n_epochs,
            lr,
            save_interval,
            seed,
            str(tmpdir),
            clip_param_threshold,
            temperature=hyperparameter,
            n_negative=10,
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
            save_interval,
            seed,
            str(tmpdir),
            clip_param_threshold,
            temperature=hyperparameter,
            alpha=1.0,
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
        ("MUTAG", 5, 2, True, "infonce", 20, 2, 0.01, 5, 0, None, 1),
        ("MUTAG", 5, 2, True, "allpairnce", 10, 2, 0.01, 5, 0, None, 1),
        ("NCI1", 5, 2, True, "triplet", 128, 2, 0.01, 5, 0, 1e-3, 1),
        ("NCI1", 10, 3, False, "nce", 256, 1, 0.001, 10, 42, None, 1),
        ("NCI1", 5, 2, True, "infonce", 256, 2, 0.01, 5, 0, 1e-3, 1),
        ("NCI1", 5, 2, True, "allpairnce", 256, 2, 0.01, 5, 0, 1e-3, 1),
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
            save_interval,
            seed,
            str(tmpdir),
            clip_param_threshold,
            margin=hyperparameter,
        )
    elif loss_name == "nce":
        cross_validation(
            dataset_name,
            k_fold,
            depth,
            normalize,
            loss_name,
            batch_size,
            n_epochs,
            lr,
            save_interval,
            seed,
            str(tmpdir),
            clip_param_threshold,
            temperature=hyperparameter,
        )
    elif loss_name == "infonce":
        cross_validation(
            dataset_name,
            k_fold,
            depth,
            normalize,
            loss_name,
            batch_size,
            n_epochs,
            lr,
            save_interval,
            seed,
            str(tmpdir),
            clip_param_threshold,
            temperature=hyperparameter,
            n_negative=10,
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
            save_interval,
            seed,
            str(tmpdir),
            clip_param_threshold,
            temperature=hyperparameter,
            alpha=1.0,
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
