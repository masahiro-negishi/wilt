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
from torch_geometric.data import Dataset  # type: ignore
from torch_geometric.datasets import TUDataset  # type: ignore

from path import DATA_DIR, RESULT_DIR  # type: ignore
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
    train_data: Dataset,
    eval_data: Dataset,
    embedding: str,
    tree: WeisfeilerLemanLabelingTree,
    seed: int,
    path: str,
    loss_name: str,
    absolute: bool,
    batch_size: int,
    n_epochs: int,
    lr: float,
    save_interval: int,
    clip_param_threshold: Optional[float],
    train_distances: torch.Tensor,
    eval_distances: torch.Tensor,
):
    """train the model with gradient descent

    Args:
        train_data (Dataset): training dataset
        eval_data (Dataset): validation dataset
        embedding (str): embedding method
        tree (WeisfeilerLemanLabelingTree): WLLT
        seed (int): random seed
        path (str): path to the directory to save the results
        loss_name (str): name of the loss function
        absolute (bool): whether to use absolute error
        batch_size (int): batch size
        n_epochs (int): number of epochs
        lr (float): learning rate
        save_interval (int): How often to save the model
        clip_param_threshold (Optional[float]): threshold for clipping the parameter
        train_distances (torch.Tensor): distance matrix for training
        eval_distances (torch.Tensor): distance matrix for evaluation
    """

    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # prepare sampler, loss function, and optimizer
    train_sampler = PairSampler(train_data, batch_size, train_distances, train=True)
    eval_sampler = PairSampler(eval_data, batch_size, eval_distances, train=False)
    if loss_name == "l1":
        loss_fn: nn.Module = torch.nn.L1Loss()
    elif loss_name == "l2":
        loss_fn = torch.nn.MSELoss()
    optimizer = Adam([tree.parameter], lr=lr)

    # save the initial model
    os.makedirs(path, exist_ok=True)
    torch.save(tree.parameter, os.path.join(path, f"model_0.pt"))

    # train the model
    train_loss_hist = []
    eval_loss_hist = []
    train_epoch_time: float = 0
    eval_epoch_time: float = 0
    if embedding == "tree":
        train_subtree_weights = torch.stack(
            [tree.calc_subtree_weights(g) for g in train_data], dim=0
        )
        eval_subtree_weights = torch.stack(
            [tree.calc_subtree_weights(g) for g in eval_data], dim=0
        )
    elif embedding == "uniform":
        train_subtree_weights = torch.rand(len(train_data), len(tree.parameter))
        eval_subtree_weights = torch.rand(len(eval_data), len(tree.parameter))
    elif embedding == "sparse-uniform":
        train_nonzero_prob = torch.count_nonzero(
            torch.stack([tree.calc_subtree_weights(g) for g in train_data], dim=0)
        ) / (len(train_data) * len(tree.parameter))
        eval_nonzero_prob = torch.count_nonzero(
            torch.stack([tree.calc_subtree_weights(g) for g in eval_data], dim=0)
        ) / (len(eval_data) * len(tree.parameter))
        train_subtree_weights = torch.where(
            torch.rand(len(train_data), len(tree.parameter)) < train_nonzero_prob,
            torch.rand(len(train_data), len(tree.parameter)),
            torch.zeros(len(train_data), len(tree.parameter)),
        )
        eval_subtree_weights = torch.where(
            torch.rand(len(eval_data), len(tree.parameter)) < eval_nonzero_prob,
            torch.rand(len(eval_data), len(tree.parameter)),
            torch.zeros(len(eval_data), len(tree.parameter)),
        )
    else:
        train_subtree_weights = torch.stack(
            [
                tree.calc_subtree_weights(g)[torch.randperm(len(tree.parameter))]
                for g in train_data
            ],
            dim=0,
        )
        eval_subtree_weights = torch.stack(
            [
                tree.calc_subtree_weights(g)[torch.randperm(len(tree.parameter))]
                for g in eval_data
            ],
            dim=0,
        )
    for epoch in range(n_epochs):
        # training
        tree.train()
        train_start = time.time()
        train_loss_sum = 0
        for i, (left_indices, right_indices, y) in enumerate(train_sampler):
            left_weights = train_subtree_weights[left_indices]
            right_weights = train_subtree_weights[right_indices]
            prediction = tree.calc_distance_between_subtree_weights(
                left_weights, right_weights
            )
            if absolute:
                loss = loss_fn(prediction, y)
            else:
                loss = loss_fn(
                    prediction / torch.clamp(y, min=1e-10), torch.ones(len(y))
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if clip_param_threshold is not None:
                tree.parameter.data = torch.clamp(
                    tree.parameter, min=clip_param_threshold
                )
            train_loss_sum += loss.item() * len(y)
            if (i + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}, Train batch {i + 1}/{len(train_sampler)}")
        train_loss_hist.append(train_loss_sum / len(train_sampler.all_pairs))
        train_end = time.time()
        train_epoch_time += train_end - train_start
        # validation
        tree.eval()
        eval_start = time.time()
        eval_loss_sum = 0
        for i, (left_indices, right_indices, y) in enumerate(eval_sampler):
            left_weights = eval_subtree_weights[left_indices]
            right_weights = eval_subtree_weights[right_indices]
            prediction = tree.calc_distance_between_subtree_weights(
                left_weights, right_weights
            )
            if absolute:
                loss = loss_fn(prediction, y)
            else:
                loss = loss_fn(
                    prediction / torch.clamp(y, min=1e-10), torch.ones(len(y))
                )
            eval_loss_sum += loss.item() * len(y)
            if (i + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}, Eval batch {i + 1}/{len(eval_sampler)}")
        eval_loss_hist.append(eval_loss_sum / len(eval_sampler.all_pairs))
        eval_end = time.time()
        eval_epoch_time += eval_end - eval_start

        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs}, Train loss: {train_loss_hist[-1]}, Eval loss: {eval_loss_hist[-1]}"
            )
        if (epoch + 1) % save_interval == 0:
            torch.save(tree.parameter, os.path.join(path, f"model_{epoch + 1}.pt"))
    train_epoch_time /= n_epochs
    eval_epoch_time /= n_epochs

    # save the loss plot
    plt.plot(train_loss_hist, label="Train")
    plt.plot(eval_loss_hist, label="Eval")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()

    # catter plots
    (
        train_abs_mean,
        train_abs_std,
        train_abs_norm_mean,
        train_abs_norm_std,
        train_rel_mean,
        train_rel_std,
        train_corr,
    ) = distance_scatter_plot(
        tree,
        train_sampler,
        train_subtree_weights,
        os.path.join(path, "scatter_train.png"),
    )
    (
        eval_abs_mean,
        eval_abs_std,
        eval_abs_norm_mean,
        eval_abs_norm_std,
        eval_rel_mean,
        eval_rel_std,
        eval_corr,
    ) = distance_scatter_plot(
        tree, eval_sampler, eval_subtree_weights, os.path.join(path, "scatter_eval.png")
    )

    # save the training information
    info = {
        "train_epoch_time": train_epoch_time,
        "eval_epoch_time": eval_epoch_time,
        "train_loss_history": train_loss_hist,
        "eval_loss_history": eval_loss_hist,
        "train_abs_mean": train_abs_mean,
        "train_abs_std": train_abs_std,
        "train_abs_norm_mean": train_abs_norm_mean,
        "train_abs_norm_std": train_abs_norm_std,
        "train_rel_mean": train_rel_mean,
        "train_rel_std": train_rel_std,
        "train_corr": train_corr,
        "eval_abs_mean": eval_abs_mean,
        "eval_abs_std": eval_abs_std,
        "eval_abs_norm_mean": eval_abs_norm_mean,
        "eval_abs_norm_std": eval_abs_norm_std,
        "eval_rel_mean": eval_rel_mean,
        "eval_rel_std": eval_rel_std,
        "eval_corr": eval_corr,
    }
    with open(os.path.join(path, "rslt.json"), "w") as f:
        json.dump(info, f)

    # save the model
    torch.save(tree.parameter, os.path.join(path, "model_final.pt"))

    return train_abs_norm_mean, eval_abs_norm_mean


def train_linear(
    train_all_data: Dataset,
    tree: WeisfeilerLemanLabelingTree,
    seed: int,
    path: str,
    n_samples: Optional[int],
    train_distances: torch.Tensor,
):
    """train the model with non-negative least squares

    Args:
        train_all_data (Dataset): training dataset
        tree (WeisfeilerLemanLabelingTree): WLLT
        seed (int): random seed
        path (str): path to the directory to save the results
        n_samples (Optional[int]): number of samples to use for training
        train_distances (torch.Tensor): distance matrix calculated by GNN
    """
    raise NotImplementedError
    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # use only some of the training data
    all_pairs = []
    for i in range(len(train_all_data)):
        for j in range(i + 1, len(train_all_data)):
            all_pairs.append((i, j))
    if n_samples is None:
        pairs = all_pairs
    else:
        pairs = random.sample(all_pairs, n_samples)

    # prepare X and y
    left_indices = torch.tensor([pair[0] for pair in pairs])
    right_indices = torch.tensor([pair[1] for pair in pairs])
    train_subtree_weights = torch.stack(
        [tree.calc_subtree_weights(g) for g in train_all_data], dim=0
    )
    X = torch.abs(
        train_subtree_weights[left_indices] - train_subtree_weights[right_indices]
    )
    zero_column = torch.sum(X, dim=0) == 0
    X = X[:, ~zero_column]
    y = train_distances[left_indices, right_indices]

    # train the model
    converged = False
    maxiter = -1
    for maxiter_cand in [100, 1000, 10000, 100000]:
        maxiter = maxiter_cand
        try:
            train_start = time.time()
            weight, _ = nnls(X, y, maxiter=maxiter)
            train_end = time.time()
            train_time = train_end - train_start
            converged = True
            break
        except:
            train_end = time.time()
            train_time = train_end - train_start
            continue

    # display prediction
    os.makedirs(path, exist_ok=True)
    if converged:
        predicted = X @ weight
        print(torch.abs(predicted - y).mean(), torch.abs(predicted - y).std())

    # save the training information
    info = {
        "train_time": train_time,
        "converged": converged,
        "maxiter": maxiter,
    }
    with open(os.path.join(path, "rslt.json"), "w") as f:
        json.dump(info, f)

    # save the model
    if converged:
        torch.save(torch.from_numpy(weight), os.path.join(path, "model_final.pt"))


def cross_validation(
    dataset_name: str,
    embedding: str,
    k_fold: int,
    depth: int,
    normalize: bool,
    seed: int,
    gnn: str,
    gnn_distance: str,
    approach: str,
    path: str,
    **kwargs,
):
    """train the model with cross validation

    Args:
        dataset_name (str): dataset name
        embedding (str): embedding method
        k_fold (int): number of splits
        depth (int): number of layers in the WLLT
        normalize (bool): whether to normalize the distribution on WLLT
        seed (int): random seed
        gnn (str): GNN model
        gnn_distance (str): distance metric for GNN embeddings
        approach (str): linear or gd
        path (str): path to the directory to save the results
        **kwargs: additional arguments
    """

    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree_start = time.time()
    if approach == "gd":
        tree = WeisfeilerLemanLabelingTree(
            data, depth, kwargs["clip_param_threshold"] is None, normalize
        )
    elif approach == "linear":
        tree = WeisfeilerLemanLabelingTree(data, depth, False, normalize)
    tree_end = time.time()
    n_samples = len(data)
    indices = np.random.RandomState(seed=seed).permutation(n_samples)

    # cross validation
    train_abs_norm_means = [-1 for _ in range(k_fold)]
    eval_abes_norm_means = [-1 for _ in range(k_fold)]
    for i in range(k_fold):
        if os.path.exists(os.path.join(path, f"fold_{i}", "rslt.json")):
            print(f"{os.path.join(path, f'fold_{i}')} already exists")
            continue
        train_indices = np.concatenate(
            (
                indices[: (i * n_samples) // k_fold],
                indices[(i + 1) * n_samples // k_fold :],
            )
        )
        eval_indices = indices[
            (i * n_samples) // k_fold : (i + 1) * n_samples // k_fold
        ]
        train_data = data[train_indices]
        eval_data = data[eval_indices]
        distances = torch.load(
            os.path.join(
                RESULT_DIR,
                "../gnn",
                f"{dataset_name}",
                f"{gnn}",
                f"{depth-1}",
                f"fold{i}",
                f"dist_{gnn_distance[-1]}.pt",
            )
        )
        train_distances = distances[train_indices][:, train_indices]
        eval_distances = distances[eval_indices][:, eval_indices]
        if approach == "linear":
            train_linear(
                train_data,
                tree,
                seed,
                os.path.join(path, f"fold_{i}"),
                kwargs["n_samples"],
                train_distances,
            )
        elif approach == "gd":
            train_abs_norm_means[i], eval_abes_norm_means[i] = train_gd(
                train_data,
                eval_data,
                embedding,
                tree,
                seed,
                os.path.join(path, f"fold_{i}"),
                kwargs["loss_name"],
                kwargs["absolute"],
                kwargs["batch_size"],
                kwargs["n_epochs"],
                kwargs["lr"],
                kwargs["save_interval"],
                kwargs["clip_param_threshold"],
                train_distances,
                eval_distances,
            )
        tree.reset_parameter()

    # save the training information
    info = {
        "dataset_name": dataset_name,
        "embedding": embedding,
        "k_fold": k_fold,
        "depth": depth,
        "normalize": normalize,
        "seed": seed,
        "gnn": gnn,
        "gnn_distance": gnn_distance,
        "tree_time": tree_end - tree_start,
    }
    if approach == "linear":
        info["approach"] = "linear"
        info["n_samples"] = kwargs["n_samples"]
    elif approach == "gd":
        info["approach"] = "gd"
        info["loss_name"] = kwargs["loss_name"]
        info["absolute"] = kwargs["absolute"]
        info["batch_size"] = kwargs["batch_size"]
        info["n_epochs"] = kwargs["n_epochs"]
        info["lr"] = kwargs["lr"]
        info["save_interval"] = kwargs["save_interval"]
        info["clip_param_threshold"] = (
            float(kwargs["clip_param_threshold"])
            if kwargs["clip_param_threshold"] is not None
            else None
        )
    info["train_abs_norm_mean"] = np.mean(train_abs_norm_means)
    info["train_abs_norm_std"] = np.std(train_abs_norm_means)
    info["eval_abs_norm_mean"] = np.mean(eval_abes_norm_means)
    info["eval_abs_norm_std"] = np.std(eval_abes_norm_means)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "info.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", choices=["MUTAG", "Mutagenicity"])
    parser.add_argument(
        "--embedding",
        choices=["tree", "uniform", "sparse-uniform", "shuffle"],
        default="tree",
    )
    parser.add_argument("--k_fold", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gnn", choices=["gcn", "gin", "gat"])
    parser.add_argument("--gnn_distance", type=str, choices=["l1", "l2"])

    subparsers = parser.add_subparsers(dest="approach")
    # linear
    linear_parser = subparsers.add_parser("linear")
    linear_parser.add_argument("--n_samples", type=int, required=False, default=None)
    # gradient descent
    gd_parser = subparsers.add_parser("gd")
    gd_parser.add_argument("--loss_name", type=str, choices=["l1", "l2"])
    gd_parser.add_argument("--absolute", action="store_true")
    gd_parser.add_argument("--batch_size", type=int)
    gd_parser.add_argument("--n_epochs", type=int)
    gd_parser.add_argument("--lr", type=float)
    gd_parser.add_argument("--save_interval", type=int)
    gd_parser.add_argument(
        "--clip_param_threshold", type=str, required=False, default=None
    )

    args = parser.parse_args()

    if args.approach == "gd" and args.clip_param_threshold is not None:
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

    if args.approach == "linear":
        kwargs["path"] = os.path.join(
            RESULT_DIR,
            args.dataset_name,
            args.gnn,
            args.gnn_distance,
            f"d{args.depth}",
            f"{norm}_n={args.n_samples}_s={args.seed}",
        )
    elif args.approach == "gd":
        kwargs["path"] = os.path.join(
            RESULT_DIR,
            args.dataset_name,
            args.gnn,
            args.gnn_distance,
            f"d{args.depth}",
            f"{norm}_l={args.loss_name}_a={args.absolute}_b={args.batch_size}_e={args.n_epochs}_lr={args.lr}_c={args.clip_param_threshold}_s={args.seed}_e={args.embedding}",
        )

    if os.path.exists(os.path.join(kwargs["path"], "info.json")):
        print(f"{kwargs['path']} already exists")
        exit()
    os.makedirs(kwargs["path"], exist_ok=True)
    cross_validation(**kwargs)
