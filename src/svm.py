import matplotlib.pyplot as plt  # type: ignore
import torch
from sklearn.svm import SVC  # type: ignore
from torch_geometric.data import Dataset  # type: ignore

from tree import WeisfeilerLemanLabelingTree
from utils import dataset_to_distance_matrix


def svm(
    tree: WeisfeilerLemanLabelingTree,
    train_data: Dataset,
    test_seen_data: Dataset,
    test_unseen_data: Dataset,
    gamma: float = 1.0,
) -> tuple[float, float, float | None, float | None]:
    """train and test a support vector machine classifier

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        train_data (Dataset): training data
        test_seen_data (Dataset): test data without unseen nodes on WLLT
        test_unseen_data (Dataset): test data with unseen nodes on WLLT
        gamma (float, optional): gamma parameter of the kernel. Defaults to 1.0.

    Returns:
        tuple[float, float, float, float]: train accuracy, test_accuracy, test_seen_accuracy, test_unseen_accuracy
    """
    distances_train = dataset_to_distance_matrix(tree, train_data)
    kernel_train = torch.exp(-gamma * distances_train)
    clf = SVC(kernel="precomputed")
    clf.fit(kernel_train, train_data.y)

    # train accuracy
    train_pred = clf.predict(kernel_train)
    train_accuracy = (torch.tensor(train_pred) == train_data.y).sum().item() / len(
        train_data
    )

    train_dists = torch.stack(
        [tree.calc_distribution_on_tree(graph) for graph in train_data],
        dim=0,
    )
    train_subtree_weights = torch.vmap(tree.calc_subtree_weight)(train_dists)

    # test seen accuracy
    if len(test_seen_data) > 0:
        test_seen_dists = torch.stack(
            [tree.calc_distribution_on_tree(graph) for graph in test_seen_data],
            dim=0,
        )
        test_seen_subtree_weights = torch.vmap(tree.calc_subtree_weight)(
            test_seen_dists
        )
        distances_test_seen = torch.tensor(
            [
                [
                    tree.calc_distance_between_subtree_weights(
                        test_seen_subtree_weights[i], train_subtree_weights[j]
                    ).item()
                    for j in range(len(train_data))
                ]
                for i in range(len(test_seen_data))
            ]
        )
        kernel_test_seen = torch.exp(-gamma * distances_test_seen)
        test_seen_pred = clf.predict(kernel_test_seen)
        test_seen_accuracy = (
            torch.tensor(test_seen_pred) == test_seen_data.y
        ).sum().item() / len(test_seen_data)
    else:
        test_seen_accuracy = torch.nan

    # test unseen accuracy
    if len(test_unseen_data) > 0:
        test_unseen_dists = torch.stack(
            [tree.calc_distribution_on_tree(graph) for graph in test_unseen_data],
            dim=0,
        )
        test_unseen_subtree_weights = torch.vmap(tree.calc_subtree_weight)(
            test_unseen_dists
        )
        distances_test_unseen = torch.tensor(
            [
                [
                    tree.calc_distance_between_subtree_weights(
                        test_unseen_subtree_weights[i], train_subtree_weights[j]
                    ).item()
                    for j in range(len(train_data))
                ]
                for i in range(len(test_unseen_data))
            ]
        )
        kernel_test_unseen = torch.exp(-gamma * distances_test_unseen)
        test_unseen_pred = clf.predict(kernel_test_unseen)
        test_unseen_accuracy = (
            torch.tensor(test_unseen_pred) == test_unseen_data.y
        ).sum().item() / len(test_unseen_data)
    else:
        test_unseen_accuracy = torch.nan

    # test accuracy
    if test_seen_accuracy is torch.nan:
        test_accuracy = test_unseen_accuracy
    elif test_unseen_accuracy is torch.nan:
        test_accuracy = test_seen_accuracy
    else:
        test_accuracy = (
            test_seen_accuracy * len(test_seen_data)
            + test_unseen_accuracy * len(test_unseen_data)
        ) / (len(test_seen_data) + len(test_unseen_data))

    return train_accuracy, test_accuracy, test_seen_accuracy, test_unseen_accuracy
