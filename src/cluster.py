import math

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
from sklearn.manifold import TSNE  # type: ignore
from torch_geometric.data import Dataset  # type: ignore


def tSNE(
    data: Dataset,
    ax: matplotlib.axes.Axes,
    indices: np.ndarray,
    distances: torch.Tensor,
) -> None:
    """tSNE visualization

    Args:
        data (Dataset): Dataset
        ax (matplotlib.axes.Axes): Axes to draw the visualization
        indices (np.ndarray): Indices of the data
        distances (torch.Tensor): distance matrix
    """
    embedding = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        random_state=0,
    ).fit_transform(distances)
    for i, c in enumerate(torch.unique(data.y)):
        class_indices = torch.where(data.y[indices] == c)[0]
        ax.scatter(
            embedding[class_indices, 0],
            embedding[class_indices, 1],
            color=plt.get_cmap("tab10").colors[i],
            label=f"class {c.item()}",
        )


def intra_inter_distance(
    data: Dataset,
    ax: matplotlib.axes.Axes,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    distances: torch.Tensor,
) -> None:
    """visualize distribution of intra and inter distances

    Args:
        data (Dataset): Dataset
        ax (matplotlib.axes.Axes): Axes to draw the visualization
        train_indices (np.ndarray): Indices of the training data
        eval_indices (np.ndarray): Indices of the evaluation data
        distances (torch.Tensor): distance matrix
    """
    train_distances = distances[train_indices][:, train_indices]
    eval_distances = distances[eval_indices][:, eval_indices]
    train_intra_distances = []
    train_inter_distances = []
    eval_intra_distances = []
    eval_inter_distances = []
    for i in range(len(train_distances)):
        for j in range(i + 1, len(train_distances)):
            if data[train_indices[i]].y == data[train_indices[j]].y:
                train_intra_distances.append(train_distances[i, j].item())
            else:
                train_inter_distances.append(train_distances[i, j].item())
    for i in range(len(eval_distances)):
        for j in range(i + 1, len(eval_distances)):
            if data[eval_indices[i]].y == data[eval_indices[j]].y:
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
