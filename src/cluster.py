import math

import matplotlib.pyplot as plt  # type: ignore
import torch
import torch_geometric.datasets  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.metrics import silhouette_score  # type: ignore

from tree import WeisfeilerLemanLabelingTree
from utils import dataset_to_distance_matrix


def tSNE(
    tree: WeisfeilerLemanLabelingTree, data: torch_geometric.datasets, path: str
) -> None:
    """tSNE visualization

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        data (torch_geometric.datasets): Dataset
        path (str): Path to save tSNE visualization
    """
    distances = dataset_to_distance_matrix(tree, data)
    embedding = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",
        random_state=0,
    ).fit_transform(distances)

    plt.figure(figsize=(10, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=data.y)
    plt.savefig(path)
    plt.close()


def intra_inter_distance(
    tree: WeisfeilerLemanLabelingTree, data: torch_geometric.datasets, path: str
) -> None:
    """visualize distribution of intra and inter distances

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        data (torch_geometric.datasets): Dataset
        path (str): Path to save intra and inter distance visualization
    """
    distances = dataset_to_distance_matrix(tree, data)
    intra_distances = []
    inter_distances = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i].y == data[j].y:
                intra_distances.append(distances[i, j].item())
            else:
                inter_distances.append(distances[i, j].item())

    # histogram
    left = math.floor(min(min(intra_distances), min(inter_distances)))
    right = math.ceil(max(max(intra_distances), max(inter_distances)))
    plt.figure(figsize=(10, 10))
    plt.hist(
        intra_distances,
        bins=right - left,
        range=(left, right),
        density=True,
        alpha=0.5,
        label="intra",
    )
    plt.hist(
        inter_distances,
        bins=right - left,
        range=(left, right),
        density=True,
        alpha=0.5,
        label="inter",
    )
    plt.legend()
    plt.title(
        f"ave intra distance: {sum(intra_distances) / len(intra_distances)}\nave inter distance: {sum(inter_distances) / len(inter_distances)}"
    )
    plt.savefig(path)
    plt.close()


def silhouette_coefficient(
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
    print(f"Silhouette Coefficient: {score}")
    return score
