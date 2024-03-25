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
    gamma: float = 1.0,
) -> float:
    """train and test a support vector machine classifier

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        train_data (torch_geometric.datasets): training data
        test_data (torch_geometric.datasets): test data
        gamma (float, optional): gamma parameter of the kernel. Defaults to 1.0.

    Returns:
        float: accuracy
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
    kernel_test = torch.exp(-gamma * distances_test)
    pred = clf.predict(kernel_test)
    accuracy = (torch.tensor(pred) == test_data.y).sum().item() / len(test_data)
    return accuracy


def svm_cross_validation(
    tree: WeisfeilerLemanLabelingTree,
    data: torch_geometric.datasets,
    k: int = 10,
    random_state: int = 0,
    gamma: float = 1.0,
) -> float:
    """perform k-fold cross validation of the support vector machine classifier

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        data (torch_geometric.datasets): Dataset
        k (int, optional): How many folds. Defaults to 10.
        random_state (int, optional): random seed. Defaults to 0.
        gamma (float, optional): gamma parameter of the kernel. Defaults to 1.0.

    Returns:
        float: average accuracy
    """
    n = len(data)
    torch.manual_seed(random_state)
    indices = torch.randperm(n)
    accuracies = []
    for i in range(k):
        test_indices = indices[i * (n // k) : (i + 1) * (n // k)]
        train_indices = torch.cat(
            [indices[: i * (n // k)], indices[(i + 1) * (n // k) :]]
        )
        train_data = data[train_indices]
        test_data = data[test_indices]
        accuracy = svm(tree, train_data, test_data, gamma)
        accuracies.append(accuracy)
    return sum(accuracies) / k
