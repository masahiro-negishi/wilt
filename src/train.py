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
from torch_geometric.loader import DataLoader  # type: ignore

from loss import NCELoss, TripletLoss
from path import RESULT_DIR  # type: ignore
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
    dataset_name: str,
    depth: int,
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
        depth (int): number of layers in the WLLT
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

    # prepare the dataset, WLLT, sampler, loss function, and optimizer
    data = TUDataset(root="data/TUDataset", name=dataset_name)
    tree_start = time.time()
    tree = WeisfeilerLemanLabelingTree(data, depth, clip_param_threshold is not None)
    tree_end = time.time()
    sampler = TripletSampler(data, batch_size)
    if loss_name == "triplet":
        loss_fn: nn.Module = TripletLoss(margin=kwargs[hyperparameter])
    else:
        loss_fn = NCELoss(temperature=kwargs[hyperparameter])
    tree.parameter.requires_grad = True
    optimizer = Adam([tree.parameter], lr=lr)

    # train the model
    loss_hist = []
    torch.save(tree.parameter, os.path.join(path, f"model_0.pt"))
    epoch_time: float = 0
    for epoch in range(n_epochs):
        train_start = time.time()
        loss_sum = 0
        for indices in sampler:
            dists = torch.stack(
                [tree.calc_distribution_on_tree(g) for g in data[indices]], dim=0
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
                    tree.parameter, min=-clip_param_threshold
                )
            loss_sum += loss.item()
        loss_hist.append(loss_sum / len(sampler))
        train_end = time.time()
        epoch_time += train_end - train_start
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss_hist[-1]}")
        if (epoch + 1) % save_interval == 0:
            torch.save(tree.parameter, os.path.join(path, f"model_{epoch + 1}.pt"))
    epoch_time /= n_epochs

    # save the training information
    info = {
        "dataset_name": dataset_name,
        "depth": depth,
        "loss_name": loss_name,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "lr": lr,
        "save_interval": save_interval,
        "seed": seed,
    }
    info[hyperparameter] = kwargs[hyperparameter]
    info["tree_time"] = tree_end - tree_start
    info["epoch_time"] = epoch_time
    info["loss_history"] = loss_hist
    with open(os.path.join(path, "info.json"), "w") as f:
        json.dump(info, f)

    # save the loss plot
    plt.plot(loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"{dataset_name}, d={depth},  l={loss_name}, b={batch_size}, lr={lr}, {hyperparameter}={kwargs[hyperparameter]}"
    )
    plt.savefig(os.path.join(path, "loss.png"))
    plt.yscale("log")
    plt.savefig(os.path.join(path, "loss_log.png"))
    plt.close()

    # save the model
    torch.save(tree.parameter, os.path.join(path, "model_final.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["MUTAG"])
    parser.add_argument("--depth", type=int)
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
        try:
            args.clip_param_threshold = float(args.clip_param_threshold)
        except ValueError:
            if args.clip_param_threshold == "smallest_normal":
                args.clip_param_threshold = np.finfo(np.float32).smallest_normal
            else:
                raise ValueError(
                    f"Invalid value for clip_param_threshold: {args.clip_param_threshold}"
                )
    kwargs = args.__dict__
    if args.loss_name == "triplet":
        kwargs["path"] = os.path.join(
            RESULT_DIR,
            f"{args.dataset_name}_d={args.depth}_{args.loss_name}_b={args.batch_size}_e={args.n_epochs}_lr={args.lr}_s={args.seed}_m={args.margin}_c={args.clip_param_threshold}",
        )
    else:
        kwargs["path"] = os.path.join(
            RESULT_DIR,
            f"{args.dataset_name}_d={args.depth}_{args.loss_name}_b={args.batch_size}_e={args.n_epochs}_lr={args.lr}_s={args.seed}_t={args.temperature}_c={args.clip_param_threshold}",
        )
    os.makedirs(kwargs["path"], exist_ok=True)
    train(**kwargs)
