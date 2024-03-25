import argparse
import json
import os
import random
import time
from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Sampler
from torch_geometric.data import Dataset  # type: ignore
from torch_geometric.datasets import TUDataset  # type: ignore

from loss import NCELoss, TripletLoss
from path import DATA_DIR, RESULT_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree


class TripletSampler(Sampler):
    """Sampler for triplet data

    Attributes:
        dataset (Dataset): dataset to sample from
        batch_size (int): batch size
        n_classes (int): number of classes
        n_samples (int): number of samples in the dataset
        positive_candidates (list[list[int]]): positive_candidates[c] is a list of indices of instances belonging to class c
        negative_candidates (list[list[int]]): negative_candidates[c] is a list of indices of instances not belonging to class c
        idx2pos (list[int]): where each instance is in positive_candidates
    """

    def __init__(self, dataset: Dataset, batch_size: int) -> None:
        """initialize the sampler

        Args:
            dataset (Dataset): dataset
            batch_size (int): batch size
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_classes = len(torch.unique(dataset.y))
        self.n_samples = len(dataset)
        self.positive_candidates: list[list[int]] = [[] for c in range(self.n_classes)]
        self.negative_candidates: list[list[int]] = [[] for c in range(self.n_classes)]
        self.idx2pos: list[int] = [-1 for _ in range(self.n_samples)]
        for idx, graph in enumerate(dataset):
            for c in range(self.n_classes):
                if graph.y == c:
                    self.positive_candidates[c].append(idx)
                    self.idx2pos[idx] = len(self.positive_candidates[c]) - 1
                else:
                    self.negative_candidates[c].append(idx)

    def __iter__(self):
        anchor_indices = torch.randperm(self.n_samples).tolist()
        positive_indices = [-1 for _ in range(self.n_samples)]
        for anc in anchor_indices:
            pidx = random.randint(
                0, len(self.positive_candidates[self.dataset[anc].y]) - 2
            )
            if pidx >= self.idx2pos[anc]:
                pidx += 1
            positive_indices[anc] = self.positive_candidates[self.dataset[anc].y][pidx]
        negative_indices = [
            self.negative_candidates[self.dataset[anc].y][
                random.randint(
                    0, len(self.negative_candidates[self.dataset[anc].y]) - 1
                )
            ]
            for anc in anchor_indices
        ]
        for i in range(0, self.n_samples, self.batch_size):
            yield anchor_indices[i : i + self.batch_size] + positive_indices[
                i : i + self.batch_size
            ] + negative_indices[i : i + self.batch_size]

    def __len__(self):
        return len(self.dataset) // self.batch_size


def train(
    train_data: Dataset,
    eval_data: Dataset,
    tree: WeisfeilerLemanLabelingTree,
    loss_name: str,
    batch_size: int,
    n_epochs: int,
    lr: float,
    path: str,
    save_interval: int,
    seed: int,
    clip_param_threshold: Optional[float] = None,
    **kwargs,
):
    """train the model

    Args:
        train_data (Dataset): training dataset
        eval_data (Dataset): validation dataset
        tree (WeisfeilerLemanLabelingTree): WLLT
        loss_name (str): name of the loss function
        batch_size (int): batch size
        n_epochs (int): number of epochs
        lr (float): learning rate
        path (str): path to the directory to save the results
        save_interval (int): How often to save the model
        seed (int): random seed
        clip_param_threshold (Optional[float]): threshold for clipping the parameter
        **kwargs: hyperparameter for loss function
    """
    hyperparameter = "margin" if loss_name == "triplet" else "temperature"

    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # prepare sampler, loss function, and optimizer
    train_sampler = TripletSampler(train_data, batch_size)
    eval_sampler = TripletSampler(eval_data, batch_size)
    if loss_name == "triplet":
        loss_fn: nn.Module = TripletLoss(margin=kwargs[hyperparameter])
    else:
        loss_fn = NCELoss(temperature=kwargs[hyperparameter])
    tree.parameter.requires_grad = True
    optimizer = Adam([tree.parameter], lr=lr)

    os.makedirs(path, exist_ok=True)
    torch.save(tree.parameter, os.path.join(path, f"model_0.pt"))

    # train the model
    train_loss_hist = []
    eval_loss_hist = []
    train_epoch_time: float = 0
    eval_epoch_time: float = 0
    for epoch in range(n_epochs):
        # training
        train_start = time.time()
        train_loss_sum = 0
        for indices in train_sampler:
            dists = torch.stack(
                [tree.calc_distribution_on_tree(g) for g in train_data[indices]], dim=0
            )
            subtree_weights = torch.vmap(tree.calc_subtree_weight)(dists)
            bs = len(indices) // 3
            anchors = subtree_weights[:bs]
            positives = subtree_weights[bs : 2 * bs]
            negatives = subtree_weights[2 * bs :]
            loss = loss_fn(tree, anchors, positives, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if clip_param_threshold is not None:
                tree.parameter.data = torch.clamp(
                    tree.parameter, min=clip_param_threshold
                )
            train_loss_sum += loss.item() * len(indices)
        train_loss_hist.append(train_loss_sum / len(train_data))
        train_end = time.time()
        train_epoch_time += train_end - train_start
        # validation
        eval_start = time.time()
        eval_loss_sum = 0
        for indices in eval_sampler:
            dists = torch.stack(
                [tree.calc_distribution_on_tree(g) for g in eval_data[indices]], dim=0
            )
            subtree_weights = torch.vmap(tree.calc_subtree_weight)(dists)
            bs = len(indices) // 3
            anchors = subtree_weights[:bs]
            positives = subtree_weights[bs : 2 * bs]
            negatives = subtree_weights[2 * bs :]
            loss = loss_fn(tree, anchors, positives, negatives)
            eval_loss_sum += loss.item() * len(indices)
        eval_loss_hist.append(eval_loss_sum / len(eval_data))
        eval_end = time.time()
        eval_epoch_time += eval_end - eval_start

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs}, Train loss: {train_loss_hist[-1]}, Eval loss: {eval_loss_hist[-1]}"
            )
        if (epoch + 1) % save_interval == 0:
            torch.save(tree.parameter, os.path.join(path, f"model_{epoch + 1}.pt"))
    train_epoch_time /= n_epochs
    eval_epoch_time /= n_epochs

    # save the training information
    info = {
        "train_epoch_time": train_epoch_time,
        "eval_epoch_time": eval_epoch_time,
        "train_loss_history": train_loss_hist,
        "eval_loss_history": eval_loss_hist,
    }
    with open(os.path.join(path, "rslt.json"), "w") as f:
        json.dump(info, f)

    # save the loss plot
    plt.plot(train_loss_hist, label="Train")
    plt.plot(eval_loss_hist, label="Eval")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(path, "loss.png"))
    plt.yscale("log")
    plt.savefig(os.path.join(path, "loss_log.png"))
    plt.close()

    # save the model
    torch.save(tree.parameter, os.path.join(path, "model_final.pt"))


def cross_validation(
    dataset_name: str,
    k_fold: int,
    depth: int,
    normalize: bool,
    loss_name: str,
    batch_size: int,
    n_epochs: int,
    lr: float,
    path: str,
    save_interval: int,
    seed: int,
    clip_param_threshold: Optional[float] = None,
    **kwargs,
):
    """train the model

    Args:
        dataset_name (str): dataset name
        k_fold (int): number of splits
        depth (int): number of layers in the WLLT
        normalize (bool): whether to normalize the distribution on WLLT
        loss_name (str): name of the loss function
        batch_size (int): batch size
        n_epochs (int): number of epochs
        lr (float): learning rate
        path (str): path to the directory to save the results
        save_interval (int): How often to save the model
        seed (int): random seed
        clip_param_threshold (Optional[float]): threshold for clipping the parameter
        **kwargs: hyperparameter for loss function
    """

    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree_start = time.time()
    tree = WeisfeilerLemanLabelingTree(
        data, depth, clip_param_threshold is None, normalize
    )
    tree_end = time.time()
    n_samples = len(data)
    indices = np.random.RandomState(seed=seed).permutation(n_samples)

    # cross validation
    for i in range(k_fold):
        train_data = data[
            np.concatenate(
                (
                    indices[: (i * n_samples) // k_fold],
                    indices[(i + 1) * n_samples // k_fold :],
                )
            )
        ]
        eval_data = data[
            indices[(i * n_samples) // k_fold : (i + 1) * n_samples // k_fold]
        ]
        train(
            train_data,
            eval_data,
            tree,
            loss_name,
            batch_size,
            n_epochs,
            lr,
            os.path.join(path, f"fold_{i}"),
            save_interval,
            seed,
            clip_param_threshold,
            **kwargs,
        )
        tree.reset_parameter()

    # save the training information
    info = {
        "dataset_name": dataset_name,
        "k_fold": k_fold,
        "depth": depth,
        "normalize": normalize,
        "loss_name": loss_name,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "lr": lr,
        "save_interval": save_interval,
        "seed": seed,
        "clip_param_threshold": (
            float(clip_param_threshold) if clip_param_threshold is not None else None
        ),
    }
    hyperparameter = "margin" if loss_name == "triplet" else "temperature"
    info[hyperparameter] = kwargs[hyperparameter]
    info["tree_time"] = tree_end - tree_start
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "info.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["MUTAG", "NCI1"])
    parser.add_argument("--k_fold", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--loss_name", choices=["triplet", "nce"])
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_interval", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--margin", type=float, required=False)
    parser.add_argument("--temperature", type=float, required=False)
    parser.add_argument("--clip_param_threshold", type=str, required=False)
    args = parser.parse_args()
    if args.clip_param_threshold is not None:
        if args.clip_param_threshold.lower() == "none":
            args.clip_param_threshold = None
        else:
            try:
                args.clip_param_threshold = float(args.clip_param_threshold)
            except:
                if args.clip_param_threshold == "smallest_normal":
                    args.clip_param_threshold = np.finfo(np.float32).smallest_normal
                else:
                    raise ValueError(
                        f"Invalid value for clip_param_threshold: {args.clip_param_threshold}"
                    )
    kwargs = args.__dict__
    norm = "norm" if args.normalize else "unnorm"
    if args.loss_name == "triplet":
        kwargs["path"] = os.path.join(
            RESULT_DIR,
            f"{args.dataset_name}_d={args.depth}_{norm}_{args.loss_name}_b={args.batch_size}_e={args.n_epochs}_lr={args.lr}_s={args.seed}_m={args.margin}_c={args.clip_param_threshold}",
        )
    else:
        kwargs["path"] = os.path.join(
            RESULT_DIR,
            f"{args.dataset_name}_d={args.depth}_{norm}_{args.loss_name}_b={args.batch_size}_e={args.n_epochs}_lr={args.lr}_s={args.seed}_t={args.temperature}_c={args.clip_param_threshold}",
        )
    os.makedirs(kwargs["path"], exist_ok=True)
    cross_validation(**kwargs)
