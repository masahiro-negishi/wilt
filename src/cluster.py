import matplotlib.pyplot as plt  # type: ignore
import torch
import torch_geometric.datasets  # type: ignore
from sklearn.manifold import TSNE  # type: ignore

from tree import WeisfeilerLemanLabelingTree


def tSNE(
    tree: WeisfeilerLemanLabelingTree, data: torch_geometric.datasets, path: str
) -> None:
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
