import math

import matplotlib.pyplot as plt  # type: ignore
import torch
import torch_geometric.datasets  # type: ignore
from sklearn.manifold import TSNE  # type: ignore

from tree import WeisfeilerLemanLabelingTree


def tSNE(
    tree: WeisfeilerLemanLabelingTree, data: torch_geometric.datasets, path: str
) -> None:
    """tSNE visualization

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        data (torch_geometric.datasets): Dataset
        path (str): Path to save tSNE visualization
    """
    dists = torch.stack(
        [tree._calc_distribution_on_tree(graph) for graph in data],
        dim=0,
    )
    embedding = TSNE(
        n_components=2,
        metric=lambda d0, d1: tree.calc_distance_between_dists(
            torch.from_numpy(d0), torch.from_numpy(d1)
        ).item(),
    ).fit_transform(dists)

    plt.figure(figsize=(10, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=data.y)
    plt.savefig(path)
    plt.close()


def intra_inter_distance(
    tree: WeisfeilerLemanLabelingTree, data: torch_geometric.datasets, path: str
) -> None:
    dists = torch.stack(
        [tree._calc_distribution_on_tree(graph) for graph in data],
        dim=0,
    )
    subtree_weights = torch.vmap(tree._calc_subtree_weight)(dists)
    intra_distances = []
    inter_distances = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            d = tree.calc_distance_between_subtree_weights(
                subtree_weights[i], subtree_weights[j]
            ).item()
            if data[i].y == data[j].y:
                intra_distances.append(d)
            else:
                inter_distances.append(d)

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
