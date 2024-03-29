import matplotlib.pyplot as plt  # type: ignore
import torch
from sklearn.svm import SVC  # type: ignore
from torch_geometric.data import Dataset  # type: ignore

from tree import WeisfeilerLemanLabelingTree
from utils import dataset_to_distance_matrix


def svm(
    tree: WeisfeilerLemanLabelingTree,
    train_data: Dataset,
    test_data: Dataset,
    gamma: float = 1.0,
) -> tuple[float, float]:
    """train and test a support vector machine classifier

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        train_data (Dataset): training data
        test_data (Dataset): test data
        gamma (float, optional): gamma parameter of the kernel. Defaults to 1.0.

    Returns:
        tuple[float, float]: train accuracy and test accuracy
    """
    distances_train = dataset_to_distance_matrix(tree, train_data)
    kernel_train = torch.exp(-gamma * distances_train)
    clf = SVC(kernel="precomputed")
    clf.fit(kernel_train, train_data.y)

    train_dists = torch.stack(
        [tree.calc_distribution_on_tree(graph) for graph in train_data],
        dim=0,
    )
    test_dists = torch.stack(
        [tree.calc_distribution_on_tree(graph) for graph in test_data],
        dim=0,
    )
    train_subtree_weights = torch.vmap(tree.calc_subtree_weight)(train_dists)
    test_subtree_weights = torch.vmap(tree.calc_subtree_weight)(test_dists)
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
    train_pred = clf.predict(kernel_train)
    train_accuracy = (torch.tensor(train_pred) == train_data.y).sum().item() / len(
        train_data
    )
    kernel_test = torch.exp(-gamma * distances_test)
    test_pred = clf.predict(kernel_test)
    test_accuracy = (torch.tensor(test_pred) == test_data.y).sum().item() / len(
        test_data
    )
    return train_accuracy, test_accuracy
