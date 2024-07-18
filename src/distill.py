import argparse
import json
import os
import random
import time
from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
from scipy.optimize import nnls  # type: ignore
from torch import nn
from torch.optim import Adam
from torch.utils.data import BatchSampler
from torch_geometric.data import Data, Dataset  # type: ignore
from torch_geometric.datasets import ZINC, MoleculeNet, TUDataset  # type: ignore

from path import DATA_DIR, GNN_DIR, RESULT_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree


class PairSampler(BatchSampler):
    """Sampler for pairwise data

    Attributes:
        dataset (Dataset): dataset to sample from
        batch_size (int): batch size
        distances (torch.Tensor): distance matrix for supervision
        train (bool): whether for training or not
        all_pairs (torch.Tensor): all pairs of indices
    """

    def __init__(
        self, dataset: Dataset, batch_size: int, distances: torch.Tensor, train: bool
    ) -> None:
        """initialize the sampler

        Args:
            dataset (Dataset): dataset
            batch_size (int): batch size
            distances (torch.Tensor): distance matrix for supervision
            train (bool): whether for training or not
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.distances = distances
        self.train = train
        n_samples = len(dataset)
        all_pairs = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                all_pairs.append((i, j))
        self.all_pairs = torch.tensor(all_pairs)  # (nC2, 2)

    def __iter__(self):
        if self.train:
            indices = torch.randperm(len(self.all_pairs))
        else:
            indices = torch.arange(len(self.all_pairs))
        for i in range(0, len(self.all_pairs), self.batch_size):
            batch_pairs = self.all_pairs[indices[i : i + self.batch_size]]
            yield batch_pairs[:, 0], batch_pairs[:, 1], self.distances[
                batch_pairs[:, 0], batch_pairs[:, 1]
            ]

    def __len__(self):
        return (len(self.all_pairs) + self.batch_size - 1) // self.batch_size


def distance_scatter_plot(
    tree: WeisfeilerLemanLabelingTree,
    sampler: PairSampler,
    subtree_weights: torch.Tensor,
    path: str,
) -> tuple[float, ...]:
    """scatter plot of (approximated distance, ground truth distance)

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        sampler (PairSampler): sampler for pairwise data
        subtree_weights (torch.Tensor): subtree weights
        path (str): path to save the plot
    """
    tree.eval()
    ys_list = []
    preds_list = []
    max_pair = 10000
    for left_indices, right_indices, y in sampler:
        left_weights = subtree_weights[left_indices]
        right_weights = subtree_weights[right_indices]
        prediction = tree.calc_distance_between_subtree_weights(
            left_weights, right_weights
        )
        ys_list.append(y)
        preds_list.append(prediction)
        if len(ys_list) * len(ys_list[0]) > max_pair:
            break
    ys = torch.cat(ys_list)
    preds = torch.cat(preds_list)
    abs_mean = torch.mean(torch.abs(preds - ys))
    abs_std = torch.std(torch.abs(preds - ys))
    abs_norm_mean = torch.mean(torch.abs(preds - ys) / torch.mean(ys))
    abs_norm_std = torch.std(torch.abs(preds - ys) / torch.mean(ys))
    rel_error = torch.abs(preds / ys - 1)
    rel_error = rel_error[~torch.isnan(rel_error) & ~torch.isinf(rel_error)]
    rel_mean = torch.mean(rel_error)
    rel_std = torch.std(rel_error)
    corr = torch.corrcoef(torch.stack([preds, ys], dim=0))
    plt.scatter(preds, ys)
    plt.plot(
        range(int(max(torch.max(preds).item(), torch.max(ys).item()))),
        range(int(max(torch.max(preds).item(), torch.max(ys).item()))),
        color="red",
    )
    plt.xlabel("Prediction")
    plt.ylabel("Ground truth")
    plt.title(
        "Abs: {:.3f}±{:.3f}, Abs(norm):{:.3f}±{:.3f},\n Rel: {:.3f}±{:.3f} Corr: {:.3f}".format(
            abs_mean,
            abs_std,
            abs_norm_mean,
            abs_norm_std,
            rel_mean,
            rel_std,
            corr[0, 1].item(),
        )
    )
    plt.savefig(path)
    plt.close()
    return (
        float(abs_mean.item()),
        float(abs_std.item()),
        float(abs_norm_mean.item()),
        float(abs_norm_std.item()),
        float(rel_mean.item()),
        float(rel_std.item()),
        float(corr[0, 1].item()),
    )


def train_gd(
    data: Dataset,
    embedding: str,
    tree: WeisfeilerLemanLabelingTree,
    seed: int,
    path: str,
    loss_name: str,
    absolute: bool,
    l1coeff: float,
    batch_size: int,
    n_epochs: int,
    lr: float,
    save_interval: int,
    clip_param_threshold: Optional[float],
    distances: torch.Tensor,
) -> float:
    """train the model with gradient descent

    Args:
        data (Dataset): training dataset
        embedding (str): embedding method
        tree (WeisfeilerLemanLabelingTree): WLLT
        seed (int): random seed
        path (str): path to the directory to save the results
        loss_name (str): name of the loss function
        absolute (bool): whether to use absolute error
        l1coeff (float): coefficient for L1 regularization
        batch_size (int): batch size
        n_epochs (int): number of epochs
        lr (float): learning rate
        save_interval (int): How often to save the model
        clip_param_threshold (Optional[float]): threshold for clipping the parameter
        distances (torch.Tensor): distance matrix for training

    Returns:
        float: mean of absolute normalized error
    """

    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # prepare sampler, loss function, and optimizer
    sampler = PairSampler(data, batch_size, distances, train=True)
    if loss_name == "l1":
        loss_fn: nn.Module = torch.nn.L1Loss()
    elif loss_name == "l2":
        loss_fn = torch.nn.MSELoss()
    optimizer = Adam([tree.parameter], lr=lr)

    # save the initial model
    os.makedirs(path, exist_ok=True)
    torch.save(tree.parameter, os.path.join(path, f"model_0.pt"))

    # train the model
    loss_hist = []
    epoch_time: float = 0
    if embedding == "tree":
        subtree_weights = torch.stack(
            [tree.calc_subtree_weights(g) for g in data], dim=0
        )
    elif embedding == "uniform":
        subtree_weights = torch.rand(len(data), len(tree.parameter))
    elif embedding == "sparse-uniform":
        nonzero_prob = torch.count_nonzero(
            torch.stack([tree.calc_subtree_weights(g) for g in data], dim=0)
        ) / (len(data) * len(tree.parameter))
        subtree_weights = torch.where(
            torch.rand(len(data), len(tree.parameter)) < nonzero_prob,
            torch.rand(len(data), len(tree.parameter)),
            torch.zeros(len(data), len(tree.parameter)),
        )
    else:
        subtree_weights = torch.stack(
            [
                tree.calc_subtree_weights(g)[torch.randperm(len(tree.parameter))]
                for g in data
            ],
            dim=0,
        )
    for epoch in range(n_epochs):
        # training
        tree.train()
        start = time.time()
        loss_sum = 0
        for i, (left_indices, right_indices, y) in enumerate(sampler):
            left_weights = subtree_weights[left_indices]
            right_weights = subtree_weights[right_indices]
            prediction = tree.calc_distance_between_subtree_weights(
                left_weights, right_weights
            )
            if absolute:
                loss = loss_fn(prediction, y)
            else:
                loss = loss_fn(
                    prediction / torch.clamp(y, min=1e-10), torch.ones(len(y))
                )
            loss += torch.sum(torch.abs(tree.parameter)) * l1coeff
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if clip_param_threshold is not None:
                tree.parameter.data = torch.clamp(
                    tree.parameter, min=clip_param_threshold
                )
            loss_sum += loss.item() * len(y)
            if (i + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}, Train batch {i + 1}/{len(sampler)}")
        loss_hist.append(loss_sum / len(sampler.all_pairs))
        end = time.time()
        epoch_time += end - start

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Train loss: {loss_hist[-1]}")
        if (epoch + 1) % save_interval == 0:
            torch.save(tree.parameter, os.path.join(path, f"model_{epoch + 1}.pt"))
    epoch_time /= n_epochs

    # save the loss plot
    plt.plot(loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()

    # catter plots
    (
        abs_mean,
        abs_std,
        abs_norm_mean,
        abs_norm_std,
        rel_mean,
        rel_std,
        corr,
    ) = distance_scatter_plot(
        tree,
        sampler,
        subtree_weights,
        os.path.join(path, "scatter.png"),
    )

    # save the training information
    info = {
        "epoch_time": epoch_time,
        "loss_history": loss_hist,
        "abs_mean": abs_mean,
        "abs_std": abs_std,
        "abs_norm_mean": abs_norm_mean,
        "abs_norm_std": abs_norm_std,
        "rel_mean": rel_mean,
        "rel_std": rel_std,
        "corr": corr,
    }
    with open(os.path.join(path, "rslt.json"), "w") as f:
        json.dump(info, f)

    # save the model
    torch.save(tree.parameter, os.path.join(path, "model_final.pt"))

    return abs_norm_mean


def train_wrapper(
    dataset_name: str,
    embedding: str,
    depth: int,
    normalize: bool,
    seed: int,
    gnn: str,
    gnn_distance: str,
    path: str,
    **kwargs,
):
    """wrapper function for training

    Args:
        dataset_name (str): dataset name
        embedding (str): embedding method
        depth (int): number of layers in the WLLT
        normalize (bool): whether to normalize the distribution on WLLT
        seed (int): random seed
        gnn (str): GNN model
        gnn_distance (str): distance metric for GNN embeddings
        path (str): path to the directory to save the results
        **kwargs: additional arguments
    """

    if dataset_name in ["synthetic_bin", "synthetic_mul", "synthetic_reg"]:
        data = torch.load(
            os.path.join(DATA_DIR, "synthetic", dataset_name[-3:], "dataset.pt")
        )
    elif dataset_name == "ZINC":
        data = ZINC(root=os.path.join(DATA_DIR, "ZINC"), subset=True, split="train")
    elif dataset_name in ["Lipo", "ESOL"]:
        data = MoleculeNet(root=os.path.join(DATA_DIR, dataset_name), name=dataset_name)
        converted = []
        xdict = {}
        edict = {}
        for d in data:
            newx = torch.zeros(len(d.x), 1, dtype=torch.int32)
            for i, x in enumerate(d.x):
                idx = tuple(x.tolist())
                if idx not in xdict:
                    xdict[idx] = len(xdict)
                newx[i][0] = xdict[idx]
            newe = torch.zeros(len(d.edge_attr), 1, dtype=torch.int32)
            for i, e in enumerate(d.edge_attr):
                idx = tuple(e.tolist())
                if idx not in edict:
                    edict[idx] = len(edict)
                newe[i][0] = edict[idx]
            converted.append(
                Data(x=newx, edge_attr=newe, edge_index=d.edge_index, y=d.y)
            )
        data = converted
    else:
        data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree_start = time.time()
    tree = WeisfeilerLemanLabelingTree(
        data, depth, kwargs["clip_param_threshold"] is None, normalize
    )
    tree_end = time.time()

    if os.path.exists(os.path.join(path, f"fold0", "rslt.json")):
        print(f"{os.path.join(path, f'fold0')} already exists")
        raise ValueError("Already exists")
    distances = torch.load(
        os.path.join(
            GNN_DIR,
            f"{dataset_name}",
            f"{gnn}",
            f"{depth-1}",
            f"fold0",
            f"dist_{gnn_distance}_last.pt",
        )
    ).to(torch.float32)
    train_abs_norm_mean = train_gd(
        data,
        embedding,
        tree,
        seed,
        os.path.join(path, f"fold0"),
        kwargs["loss_name"],
        kwargs["absolute"],
        kwargs["l1coeff"],
        kwargs["batch_size"],
        kwargs["n_epochs"],
        kwargs["lr"],
        kwargs["save_interval"],
        kwargs["clip_param_threshold"],
        distances,
    )
    tree.reset_parameter()

    # save the training information
    info = {
        "dataset_name": dataset_name,
        "embedding": embedding,
        "depth": depth,
        "normalize": normalize,
        "seed": seed,
        "gnn": gnn,
        "gnn_distance": gnn_distance,
        "tree_time": tree_end - tree_start,
        "loss_name": kwargs["loss_name"],
        "absolute": kwargs["absolute"],
        "l1coeff": kwargs["l1coeff"],
        "batch_size": kwargs["batch_size"],
        "n_epochs": kwargs["n_epochs"],
        "lr": kwargs["lr"],
        "save_interval": kwargs["save_interval"],
        "clip_param_threshold": (
            float(kwargs["clip_param_threshold"])
            if kwargs["clip_param_threshold"] is not None
            else None
        ),
        "train_abs_norm_mean": train_abs_norm_mean,
    }
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "info.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        choices=[
            "MUTAG",
            "Mutagenicity",
            "NCI1",
            "ENZYMES",
            "synthetic_bin",
            "synthetic_mul",
            "synthetic_reg",
            "ZINC",
            "Lipo",
            "ESOL",
        ],
    )
    parser.add_argument(
        "--embedding",
        choices=["tree", "uniform", "sparse-uniform", "shuffle"],
        default="tree",
    )
    parser.add_argument("--depth", type=int)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gnn", choices=["gcn", "gin", "gat"])
    parser.add_argument("--gnn_distance", type=str, choices=["l1", "l2"])
    parser.add_argument("--loss_name", type=str, choices=["l1", "l2"])
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument("--l1coeff", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_interval", type=int)
    parser.add_argument(
        "--clip_param_threshold", type=str, required=False, default=None
    )

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

    kwargs["path"] = os.path.join(
        RESULT_DIR,
        args.dataset_name,
        args.gnn,
        args.gnn_distance,
        f"d{args.depth}",
        f"{norm}_l={args.loss_name}_a={args.absolute}_l1={args.l1coeff}_b={args.batch_size}_e={args.n_epochs}_lr={args.lr}_c={args.clip_param_threshold}_s={args.seed}_e={args.embedding}",
    )

    if os.path.exists(os.path.join(kwargs["path"], "info.json")):
        print(f"{kwargs['path']} already exists")
        exit()
    os.makedirs(kwargs["path"], exist_ok=True)
    train_wrapper(**kwargs)
