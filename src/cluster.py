import math

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import torch_geometric.datasets  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.metrics import silhouette_score  # type: ignore

from tree import WeisfeilerLemanLabelingTree
from utils import dataset_to_distance_matrix


def tSNE(
    tree: WeisfeilerLemanLabelingTree,
    data: torch_geometric.datasets,
    ax: matplotlib.axes.Axes,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> None:
    """tSNE visualization

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        data (torch_geometric.datasets): Dataset
        ax (matplotlib.axes.Axes): Axes to draw the visualization
        train_indices (np.ndarray): Indices of the training data
        eval_indices (np.ndarray): Indices of the evaluation data
    """
    distances = dataset_to_distance_matrix(tree, data[train_indices])
    embedding = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        random_state=0,
    ).fit_transform(distances)
    for i, c in enumerate(torch.unique(data.y)):
        indices = torch.where(data.y[train_indices] == c)[0]
        ax.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            color=plt.get_cmap("tab10").colors[i],
            label=f"class {c.item()}",
        )


def intra_inter_distance(
    tree: WeisfeilerLemanLabelingTree,
    data: torch_geometric.datasets,
    ax: matplotlib.axes.Axes,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> None:
    """visualize distribution of intra and inter distances

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        data (torch_geometric.datasets): Dataset
        ax (matplotlib.axes.Axes): Axes to draw the visualization
        train_indices (np.ndarray): Indices of the training data
        eval_indices (np.ndarray): Indices of the evaluation data
    """
    train_distances = dataset_to_distance_matrix(tree, data[train_indices])
    eval_distances = dataset_to_distance_matrix(tree, data[eval_indices])
    train_intra_distances = []
    train_inter_distances = []
    eval_intra_distances = []
    eval_inter_distances = []
    for i in range(len(train_distances)):
        for j in range(i + 1, len(train_distances)):
            if data[i].y == data[j].y:
                train_intra_distances.append(train_distances[i, j].item())
            else:
                train_inter_distances.append(train_distances[i, j].item())
    for i in range(len(eval_distances)):
        for j in range(i + 1, len(eval_distances)):
            if data[i].y == data[j].y:
                eval_intra_distances.append(eval_distances[i, j].item())
            else:
                eval_inter_distances.append(eval_distances[i, j].item())

    # histogram
    left = math.floor(
        min(
            min(train_intra_distances),
            min(train_inter_distances),
            min(eval_intra_distances),
            min(eval_inter_distances),
        )
    )
    right = math.ceil(
        max(
            max(train_intra_distances),
            max(train_inter_distances),
            max(eval_intra_distances),
            max(eval_inter_distances),
        )
    )
    train_intra_hist, _, _ = ax.hist(
        [train_intra_distances, train_inter_distances],
        bins=10,
        range=(left, right),
        density=True,
        label=["train intra", "train inter"],
    )
    ax.hist(
        [eval_intra_distances, eval_inter_distances],
        bins=10,
        range=(left, right),
        density=True,
        bottom=np.max(train_intra_hist) * 1.1,
        label=["eval intra", "eval inter"],
    )


def silhouette(
    tree: WeisfeilerLemanLabelingTree, data: torch_geometric.datasets
) -> float:
    """Silhouette coefficient

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        data (torch_geometric.datasets): Dataset

    Returns:
        float: [-1, 1]. The best value is 1 and the worst value is -1.
    """
    distances = dataset_to_distance_matrix(tree, data)
    score = silhouette_score(
        distances, labels=data.y, metric="precomputed", random_state=0
    )
    return score
