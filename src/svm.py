import torch
from sklearn.svm import SVC  # type: ignore
from torch_geometric.data import Dataset  # type: ignore


def svm(
    data: Dataset,
    train_indices: torch.Tensor,
    test_seen_indices: torch.Tensor,
    test_unseen_indices: torch.Tensor,
    distances: torch.Tensor,
    gamma: float = 1.0,
) -> tuple[float, float, float | None, float | None]:
    """train and test a support vector machine classifier

    Args:
        data (Dataset): dataset
        train_indices (torch.Tensor): indices for training data
        test_seen_indices (torch.Tensor): indices for test data without unseen nodes on WLLT
        test_unseen_indices (torch.Tensor): indices for test data with unseen nodes on WLLT
        distances (torch.Tensor): distance matrix
        gamma (float, optional): gamma parameter of the kernel. Defaults to 1.0.

    Returns:
        tuple[float, float, float, float]: train accuracy, test_accuracy, test_seen_accuracy, test_unseen_accuracy
    """
    train_data = data[train_indices]
    distances_train = distances[train_indices][:, train_indices]
    kernel_train = torch.exp(-gamma * distances_train)
    clf = SVC(kernel="precomputed")
    clf.fit(kernel_train, train_data.y)

    # train accuracy
    train_pred = clf.predict(kernel_train)
    train_accuracy = (torch.tensor(train_pred) == train_data.y).sum().item() / len(
        train_data
    )

    # test seen accuracy
    if len(test_seen_indices) > 0:
        test_seen_data = data[test_seen_indices]
        distances_test_seen = distances[test_seen_indices][:, train_indices]
        kernel_test_seen = torch.exp(-gamma * distances_test_seen)
        test_seen_pred = clf.predict(kernel_test_seen)
        test_seen_accuracy = (
            torch.tensor(test_seen_pred) == test_seen_data.y
        ).sum().item() / len(test_seen_data)
    else:
        test_seen_accuracy = torch.nan

    # test unseen accuracy
    if len(test_unseen_indices) > 0:
        test_unseen_data = data[test_unseen_indices]
        distances_test_unseen = distances[test_unseen_indices][:, train_indices]
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
