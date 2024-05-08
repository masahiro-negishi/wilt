import argparse
import json
import os
import random
import time
from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch

# from scipy.linalg import lstsq  # type: ignore
from scipy.optimize import nnls  # type: ignore
from torch import nn
from torch.optim import Adam
from torch.utils.data import BatchSampler, RandomSampler
from torch_geometric.data import Dataset  # type: ignore
from torch_geometric.datasets import TUDataset  # type: ignore

from loss import AllPairNCELoss, InfoNCELoss, KnnNCELoss, NCELoss, TripletLoss
from path import DATA_DIR, RESULT_DIR  # type: ignore
from tree import WeisfeilerLemanLabelingTree


class TripletSampler(BatchSampler):
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
        self.positive_candidates: list[list[int]] = [[] for _ in range(self.n_classes)]
        self.negative_candidates: list[list[int]] = [[] for _ in range(self.n_classes)]
        self.idx2pos: list[int] = [
            -1 for _ in range(self.n_samples)
        ]  # position of each instance in positive_candidates
        for idx, graph in enumerate(dataset):
            for c in range(self.n_classes):
                if graph.y == c:
                    self.positive_candidates[c].append(idx)
                    self.idx2pos[idx] = len(self.positive_candidates[c]) - 1
                else:
                    self.negative_candidates[c].append(idx)

    def __iter__(self):
        anchor_indices = torch.randperm(self.n_samples)
        positive_indices = [-1 for _ in range(self.n_samples)]
        for i, anc in enumerate(anchor_indices):
            pidx = random.randint(
                0, len(self.positive_candidates[self.dataset[anc].y]) - 2
            )
            if pidx >= self.idx2pos[anc]:
                pidx += 1
            positive_indices[i] = self.positive_candidates[self.dataset[anc].y][pidx]
        negative_indices = [
            self.negative_candidates[self.dataset[anc].y][
                random.randint(
                    0, len(self.negative_candidates[self.dataset[anc].y]) - 1
                )
            ]
            for anc in anchor_indices
        ]
        positive_indices = torch.tensor(positive_indices)
        negative_indices = torch.tensor(negative_indices)
        for i in range(0, self.n_samples, self.batch_size):
            yield anchor_indices[i : i + self.batch_size], positive_indices[
                i : i + self.batch_size
            ], negative_indices[i : i + self.batch_size]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class NPlusTwoSampler(BatchSampler):
    """Sampler for anchor, positive, and multiple negative data

    Attributes:
        dataset (Dataset): dataset to sample from
        batch_size (int): batch size
        n_negative (int): number of negative samples
        n_classes (int): number of classes
        n_samples (int): number of samples in the dataset
        positive_candidates (list[list[int]]): positive_candidates[c] is a list of indices of instances belonging to class c
        negative_candidates (list[list[int]]): negative_candidates[c] is a list of indices of instances not belonging to class c
        idx2pos (list[int]): where each instance is in positive_candidates
    """

    def __init__(self, dataset: Dataset, batch_size: int, n_negative: int) -> None:
        """initialize the sampler

        Args:
            dataset (Dataset): dataset
            batch_size (int): batch size
            n_negative (int): number of negative samples
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_negative = n_negative
        self.n_classes = len(torch.unique(dataset.y))
        self.n_samples = len(dataset)
        self.positive_candidates: list[list[int]] = [[] for _ in range(self.n_classes)]
        self.negative_candidates: list[list[int]] = [[] for _ in range(self.n_classes)]
        self.idx2pos: list[int] = [
            -1 for _ in range(self.n_samples)
        ]  # position of each instance in positive_candidates
        for idx, graph in enumerate(dataset):
            for c in range(self.n_classes):
                if graph.y == c:
                    self.positive_candidates[c].append(idx)
                    self.idx2pos[idx] = len(self.positive_candidates[c]) - 1
                else:
                    self.negative_candidates[c].append(idx)

    def __iter__(self):
        anchor_indices = torch.randperm(self.n_samples)
        positive_indices = [-1 for _ in range(self.n_samples)]
        for i, anc in enumerate(anchor_indices):
            pidx = random.randint(
                0, len(self.positive_candidates[self.dataset[anc].y]) - 2
            )
            if pidx >= self.idx2pos[anc]:
                pidx += 1
            positive_indices[i] = self.positive_candidates[self.dataset[anc].y][pidx]
        negative_indices = [
            [
                self.negative_candidates[self.dataset[anc].y][
                    random.randint(
                        0, len(self.negative_candidates[self.dataset[anc].y]) - 1
                    )
                ]
                for _ in range(
                    self.n_negative
                )  # the same negative sample can be selected multiple times
            ]
            for anc in anchor_indices
        ]
        positive_indices = torch.tensor(positive_indices)
        negative_indices = torch.tensor(negative_indices)
        for i in range(0, self.n_samples, self.batch_size):
            yield anchor_indices[i : i + self.batch_size], positive_indices[
                i : i + self.batch_size
            ], negative_indices[i : i + self.batch_size]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def train(
    train_data: Dataset,
    eval_data: Dataset,
    tree: WeisfeilerLemanLabelingTree,
    seed: int,
    path: str,
    loss_name: str,
    batch_size: int,
    n_epochs: int,
    lr: float,
    save_interval: int,
    clip_param_threshold: Optional[float] = None,
    **kwargs,
):
    """train the model

    Args:
        train_data (Dataset): training dataset
        eval_data (Dataset): validation dataset
        tree (WeisfeilerLemanLabelingTree): WLLT
        seed (int): random seed
        path (str): path to the directory to save the results
        loss_name (str): name of the loss function
        batch_size (int): batch size
        n_epochs (int): number of epochs
        lr (float): learning rate
        save_interval (int): How often to save the model
        clip_param_threshold (Optional[float]): threshold for clipping the parameter
        **kwargs: hyperparameters for loss function
    """

    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # prepare sampler, loss function, and optimizer
    if loss_name in ["triplet", "nce"]:
        train_sampler: BatchSampler = TripletSampler(train_data, batch_size)
        eval_sampler: BatchSampler = TripletSampler(eval_data, batch_size)
        if loss_name == "triplet":
            loss_fn: nn.Module = TripletLoss(margin=kwargs["margin"])
        else:
            loss_fn = NCELoss(temperature=kwargs["temperature"])
    elif loss_name == "infonce":
        train_sampler = NPlusTwoSampler(train_data, batch_size, kwargs["n_negative"])
        eval_sampler = NPlusTwoSampler(eval_data, batch_size, kwargs["n_negative"])
        loss_fn = InfoNCELoss(temperature=kwargs["temperature"])
    elif loss_name in ["allpairnce", "knnnce"]:
        train_sampler = BatchSampler(
            RandomSampler(train_data), batch_size, drop_last=False
        )
        eval_sampler = BatchSampler(
            RandomSampler(eval_data), batch_size, drop_last=False
        )
        if loss_name == "allpairnce":
            loss_fn = AllPairNCELoss(
                temperature=kwargs["temperature"], alpha=kwargs["alpha"]
            )
        else:
            assert kwargs["n_neighbors"] <= batch_size
            loss_fn = KnnNCELoss(
                temperature=kwargs["temperature"],
                alpha=kwargs["alpha"],
                n_neighbors=kwargs["n_neighbors"],
            )
    optimizer = Adam([tree.parameter], lr=lr)

    # save the initial model
    os.makedirs(path, exist_ok=True)
    torch.save(tree.parameter, os.path.join(path, f"model_0.pt"))

    # train the model
    train_loss_hist = []
    eval_loss_hist = []
    train_epoch_time: float = 0
    eval_epoch_time: float = 0
    train_subtree_weights = torch.stack(
        [tree.calc_subtree_weights(g) for g in train_data], dim=0
    )
    eval_subtree_weights = torch.stack(
        [tree.calc_subtree_weights(g) for g in eval_data], dim=0
    )
    for epoch in range(n_epochs):
        # training
        tree.train()
        train_start = time.time()
        train_loss_sum = 0
        if loss_name in ["triplet", "nce", "infonce"]:
            for anchor_indices, positive_indices, negative_indices in train_sampler:
                anchors = train_subtree_weights[anchor_indices]
                positives = train_subtree_weights[positive_indices]
                negatives = train_subtree_weights[negative_indices]
                loss = loss_fn(tree, anchors, positives, negatives)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if clip_param_threshold is not None:
                    tree.parameter.data = torch.clamp(
                        tree.parameter, min=clip_param_threshold
                    )
                train_loss_sum += loss.item() * len(anchors)
        else:
            for indices in train_sampler:
                loss = loss_fn(
                    tree, train_subtree_weights[indices], train_data[indices].y
                )
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
        tree.eval()
        eval_start = time.time()
        eval_loss_sum = 0
        if loss_name in ["triplet", "nce", "infonce"]:
            for anchor_indices, positive_indices, negative_indices in eval_sampler:
                anchors = eval_subtree_weights[anchor_indices]
                positives = eval_subtree_weights[positive_indices]
                negatives = eval_subtree_weights[negative_indices]
                loss = loss_fn(tree, anchors, positives, negatives)
                eval_loss_sum += loss.item() * len(anchors)
        else:
            for indices in eval_sampler:
                loss = loss_fn(
                    tree, eval_subtree_weights[indices], eval_data[indices].y
                )
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


def train_linear(
    train_all_data: Dataset,
    tree: WeisfeilerLemanLabelingTree,
    seed: int,
    path: str,
    same_label: int,
    diff_label: int,
    n_samples: Optional[int],
):
    """train the model

    Args:
        train_all_data (Dataset): training dataset
        tree (WeisfeilerLemanLabelingTree): WLLT
        seed (int): random seed
        path (str): path to the directory to save the results
        same_label (int): target distance value for a pair of instances with the same label
        diff_label (int): target distance value for a pair of instances with different labels
        n_samples (int): number of samples to use for training
    """
    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # use only some of the training data
    classes, _ = torch.sort(torch.unique(train_all_data.y))
    assert len(classes) == 2  # only binary classification is supported
    indices_neg = torch.where(train_all_data.y == classes[0])[0]
    indices_pos = torch.where(train_all_data.y == classes[1])[0]
    if n_samples is None:
        n_samples = 2 * min(len(indices_neg), len(indices_pos))
    train_neg = train_all_data[indices_neg[: n_samples // 2]]
    train_pos = train_all_data[indices_pos[: n_samples // 2]]

    # calculate distances
    train_subtree_weights_neg = torch.stack(
        [tree.calc_subtree_weights(g) for g in train_neg], dim=0
    )
    train_subtree_weights_pos = torch.stack(
        [tree.calc_subtree_weights(g) for g in train_pos], dim=0
    )
    train_subtree_weights = torch.cat(
        (train_subtree_weights_neg, train_subtree_weights_pos), dim=0
    )
    l1_neg = torch.abs(
        train_subtree_weights[: n_samples // 2]
        - train_subtree_weights[: n_samples // 2].unsqueeze(1)
    )  # n_samples/2 * n_samples/2 * n_nodes
    l1_pos = torch.abs(
        train_subtree_weights[n_samples // 2 :]
        - train_subtree_weights[n_samples // 2 :].unsqueeze(1)
    )  # n_samples/2 * n_samples/2 * n_nodes
    l1_cross = torch.abs(
        train_subtree_weights[: n_samples // 2]
        - train_subtree_weights[n_samples // 2 :].unsqueeze(1)
    )  # n_samples/2 * n_samples/2 * n_nodes
    indices_upper_triangular = torch.triu_indices(n_samples // 2, n_samples // 2, 1)
    l1_neg_flat = l1_neg[indices_upper_triangular[0], indices_upper_triangular[1], :]
    l1_pos_flat = l1_pos[indices_upper_triangular[0], indices_upper_triangular[1], :]
    l1_cross_flat = l1_cross.reshape(-1, l1_cross.size(-1))
    X = torch.cat((l1_neg_flat, l1_pos_flat, l1_cross_flat), dim=0)
    y = torch.cat(
        (
            torch.full((l1_neg_flat.size(0),), same_label),
            torch.full((l1_pos_flat.size(0),), same_label),
            torch.full((l1_cross_flat.size(0),), diff_label),
        ),
        dim=0,
    )

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

    # save the training information
    os.makedirs(path, exist_ok=True)
    info = {
        "train_time": train_time,
        "converged": converged,
        "maxiter": maxiter,
    }
    with open(os.path.join(path, "rslt.json"), "w") as f:
        json.dump(info, f)
    # save the model
    if converged:
        torch.save(weight, os.path.join(path, "model_final.pt"))


def cross_validation(
    dataset_name: str,
    k_fold: int,
    depth: int,
    normalize: bool,
    seed: int,
    approach: str,
    path: str,
    **kwargs,
):
    """train the model

    Args:
        dataset_name (str): dataset name
        k_fold (int): number of splits
        depth (int): number of layers in the WLLT
        normalize (bool): whether to normalize the distribution on WLLT
        seed (int): random seed
        approach (str): contrastive or linear
        path (str): path to the directory to save the results
        **kwargs: hyperparameters for loss function
    """

    data = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    tree_start = time.time()
    if approach == "contrastive":
        tree = WeisfeilerLemanLabelingTree(
            data, depth, kwargs["clip_param_threshold"] is None, normalize
        )
    elif approach == "linear":
        tree = WeisfeilerLemanLabelingTree(data, depth, False, normalize)
    tree_end = time.time()
    n_samples = len(data)
    indices = np.random.RandomState(seed=seed).permutation(n_samples)

    # cross validation
    for i in range(k_fold):
        if os.path.exists(os.path.join(path, f"fold_{i}", "rslt.json")):
            print(f"{os.path.join(path, f'fold_{i}')} already exists")
            continue
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
        if approach == "contrastive":
            train(
                train_data,
                eval_data,
                tree,
                seed,
                os.path.join(path, f"fold_{i}"),
                **kwargs,
            )
        elif approach == "linear":
            train_linear(
                train_data,
                tree,
                seed,
                os.path.join(path, f"fold_{i}"),
                **kwargs,
            )
        tree.reset_parameter()

    # save the training information
    info = {
        "dataset_name": dataset_name,
        "k_fold": k_fold,
        "depth": depth,
        "normalize": normalize,
        "seed": seed,
    }
    if approach == "contrastive":
        info["approach"] = "contrastive"
        info["loss_name"] = kwargs["loss_name"]
        info["batch_size"] = kwargs["batch_size"]
        info["n_epochs"] = kwargs["n_epochs"]
        info["lr"] = kwargs["lr"]
        info["save_interval"] = kwargs["save_interval"]
        info["clip_param_threshold"] = (
            float(kwargs["clip_param_threshold"])
            if kwargs["clip_param_threshold"] is not None
            else None
        )
        if kwargs["loss_name"] == "triplet":
            info["margin"] = kwargs["margin"]
        elif kwargs["loss_name"] == "nce":
            info["temperature"] = kwargs["temperature"]
        elif kwargs["loss_name"] == "infonce":
            info["temperature"] = kwargs["temperature"]
            info["n_negative"] = kwargs["n_negative"]
        elif kwargs["loss_name"] == "allpairnce":
            info["temperature"] = kwargs["temperature"]
            info["alpha"] = kwargs["alpha"]
        elif kwargs["loss_name"] == "knnnce":
            info["temperature"] = kwargs["temperature"]
            info["alpha"] = kwargs["alpha"]
            info["n_neighbors"] = kwargs["n_neighbors"]
    elif approach == "linear":
        info["approach"] = "linear"
        info["same_label"] = kwargs["same_label"]
        info["diff_label"] = kwargs["diff_label"]
        info["n_samples"] = kwargs["n_samples"]
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
    parser.add_argument("--seed", type=int)

    subparsers = parser.add_subparsers(dest="approach")
    # contrstive
    contrastive_parser = subparsers.add_parser("contrastive")
    contrastive_parser.add_argument(
        "--loss_name",
        choices=["triplet", "nce", "infonce", "allpairnce", "knnnce"],
        required=False,
    )
    contrastive_parser.add_argument("--batch_size", type=int, required=False)
    contrastive_parser.add_argument("--n_epochs", type=int, required=False)
    contrastive_parser.add_argument("--lr", type=float, required=False)
    contrastive_parser.add_argument("--save_interval", type=int, required=False)
    contrastive_parser.add_argument("--margin", type=float, required=False)
    contrastive_parser.add_argument("--temperature", type=float, required=False)
    contrastive_parser.add_argument("--n_negative", type=int, required=False)
    contrastive_parser.add_argument("--alpha", type=float, required=False)
    contrastive_parser.add_argument("--n_neighbors", type=int, required=False)
    contrastive_parser.add_argument("--clip_param_threshold", type=str, required=False)
    # linear
    linear_parser = subparsers.add_parser("linear")
    linear_parser.add_argument("--same_label", type=int)
    linear_parser.add_argument("--diff_label", type=int)
    linear_parser.add_argument("--n_samples", type=int, required=False, default=None)

    args = parser.parse_args()

    if args.approach == "contrastive" and args.clip_param_threshold is not None:
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
    if args.approach == "contrastive":
        if args.loss_name == "triplet":
            kwargs["path"] = os.path.join(
                RESULT_DIR,
                args.dataset_name,
                args.loss_name,
                f"d{args.depth}",
                f"{norm}_b={args.batch_size}_e={args.n_epochs}_lr={args.lr}_s={args.seed}_m={args.margin}_c={args.clip_param_threshold}",
            )
        elif args.loss_name == "nce":
            kwargs["path"] = os.path.join(
                RESULT_DIR,
                args.dataset_name,
                args.loss_name,
                f"d{args.depth}",
                f"{norm}_b={args.batch_size}_e={args.n_epochs}_lr={args.lr}_s={args.seed}_t={args.temperature}_c={args.clip_param_threshold}",
            )
        elif args.loss_name == "infonce":
            kwargs["path"] = os.path.join(
                RESULT_DIR,
                args.dataset_name,
                args.loss_name,
                f"d{args.depth}",
                f"{norm}_b={args.batch_size}_e={args.n_epochs}_lr={args.lr}_s={args.seed}_t={args.temperature}_n={args.n_negative}_c={args.clip_param_threshold}",
            )
        elif args.loss_name == "allpairnce":
            kwargs["path"] = os.path.join(
                RESULT_DIR,
                args.dataset_name,
                args.loss_name,
                f"d{args.depth}",
                f"{norm}_b={args.batch_size}_e={args.n_epochs}_lr={args.lr}_s={args.seed}_t={args.temperature}_a={args.alpha}_c={args.clip_param_threshold}",
            )
        elif args.loss_name == "knnnce":
            kwargs["path"] = os.path.join(
                RESULT_DIR,
                args.dataset_name,
                args.loss_name,
                f"d{args.depth}",
                f"{norm}_b={args.batch_size}_e={args.n_epochs}_lr={args.lr}_s={args.seed}_t={args.temperature}_a={args.alpha}_nei={args.n_neighbors}_c={args.clip_param_threshold}",
            )
        else:
            raise ValueError(f"Invalid loss name: {args.loss_name}")
    elif args.approach == "linear":
        kwargs["path"] = os.path.join(
            RESULT_DIR,
            args.dataset_name,
            "linear",
            f"d{args.depth}",
            f"{norm}_sl={args.same_label}_dl={args.diff_label}_n={args.n_samples}_s={args.seed}",
        )
    else:
        raise ValueError(f"Invalid approach: {args.approach}")
    if os.path.exists(os.path.join(kwargs["path"], "info.json")):
        print(f"{kwargs['path']} already exists")
        exit()
    os.makedirs(kwargs["path"], exist_ok=True)
    cross_validation(**kwargs)
