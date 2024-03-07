import torch
import torch_geometric.datasets  # type: ignore

from tree import WeisfeilerLemanLabelingTree


def dataset_to_distance_matrix(
    tree: WeisfeilerLemanLabelingTree, data: torch_geometric.datasets
) -> torch.Tensor:
    """convert dataset to distance matrix

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        data (torch_geometric.datasets): Dataset

    Returns:
        torch.Tensor: distance matrix
    """
    dists = torch.stack(
        [tree.calc_distribution_on_tree(graph) for graph in data],
        dim=0,
    )
    subtree_weights = torch.vmap(tree.calc_subtree_weight)(dists)
    distances = torch.tensor(
        [
            [
                tree.calc_distance_between_subtree_weights(
                    subtree_weights[i], subtree_weights[j]
                ).item()
                for j in range(len(data))
            ]
            for i in range(len(data))
        ]
    )
    return distances
