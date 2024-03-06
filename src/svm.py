import matplotlib.pyplot as plt  # type: ignore
import torch
import torch_geometric.datasets  # type: ignore
from sklearn.svm import SVC  # type: ignore

from tree import WeisfeilerLemanLabelingTree
from utils import dataset_to_distance_matrix


def svm(
    tree: WeisfeilerLemanLabelingTree,
    train_data: torch_geometric.datasets,
    test_data: torch_geometric.datasets,
) -> float:
    distances_train = dataset_to_distance_matrix(tree, train_data)
    kernel_train = torch.exp(-1 * distances_train)
    clf = SVC(kernel="precomputed")
    clf.fit(kernel_train, train_data.y)

    train_dists = torch.stack(
        [tree._calc_distribution_on_tree(graph) for graph in train_data],
        dim=0,
    )
    test_dists = torch.stack(
        [tree._calc_distribution_on_tree(graph) for graph in test_data],
        dim=0,
    )
    train_subtree_weights = torch.vmap(tree._calc_subtree_weight)(train_dists)
    test_subtree_weights = torch.vmap(tree._calc_subtree_weight)(test_dists)
    distances_test = torch.tensor(
        [
            [
                tree.calc_distance_between_subtree_weights(
                    test_subtree_weights[i], train_subtree_weights[j]
                ).item()
                for j in range(len(train_data))
            ]
            for i in range(len(test_data))
        ]
    )
    kernel_test = torch.exp(-1 * distances_test)
    pred = clf.predict(kernel_test)
    accuracy = (torch.tensor(pred) == test_data.y).sum().item() / len(test_data)
    print(f"Accuracy: {accuracy}")
    return accuracy
