import json
import os
import random

import matplotlib.pyplot as plt  # type: ignore
import torch
from torch.optim import Adam
from torch.utils.data import Sampler
from torch_geometric.data import Dataset  # type: ignore
from torch_geometric.datasets import TUDataset  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore

from loss import TripletLoss
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
    batch_size: int,
    n_epochs: int,
    lr: float,
    path: str,
    **kwargs,
):
    """train the model

    Args:
        dataset_name (str): dataset name
        depth (int): number of layers in the WLLT
        batch_size (int): batch size
        n_epochs (int): number of epochs
        lr (float): learning rate
        path (str): path to the directory to save the results
        **kwargs: hyperparameters for loss function
    """
    # prepare the dataset, WLLT, sampler, loss function, and optimizer
    data = TUDataset(root="data/TUDataset", name=dataset_name)
    tree = WeisfeilerLemanLabelingTree(data, depth)
    sampler = TripletSampler(data, batch_size)
    loss_fn = TripletLoss(margin=kwargs["margin"], n_classes=len(torch.unique(data.y)))
    tree.weight.requires_grad = True
    optimizer = Adam([tree.weight], lr=lr)

    # train the model
    loss_hist = []
    for epoch in range(n_epochs):
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
            loss_sum += loss.item()
        loss_hist.append(loss_sum / len(sampler))
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss_hist[-1]}")

    # save the training information
    info = {
        "dataset_name": dataset_name,
        "depth": depth,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "lr": lr,
        "loss_history": loss_hist,
    }
    for key, val in kwargs.items():
        info[key] = val
    with open(os.path.join(path, "info.json"), "w") as f:
        json.dump(info, f)

    # save the loss plot
    plt.plot(loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"{dataset_name}, d={depth}, b={batch_size}, lr={lr}, m={kwargs['margin']}"
    )
    plt.savefig(os.path.join(path, "loss.png"))
    plt.yscale("log")
    plt.savefig(os.path.join(path, "loss_log.png"))
    plt.close()

    # save the model
    torch.save(tree.weight, os.path.join(path, "model.pt"))
