import torch
from torch import nn

from tree import WeisfeilerLemanLabelingTree


class TripletLoss(nn.Module):
    """Triple loss function

    Attributes:
        margin (float): Margin
    """

    def __init__(self, margin: float):
        """initialize triplet loss function

        Args:
            margin (float): hyperparameter for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        tree: WeisfeilerLemanLabelingTree,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """calculate triplet loss

        Args:
            tree (WeisfeilerLemanLabelingTree): WLLT
            anchors (torch.Tensor): anchor samples
            positives (torch.Tensor): positive samples
            negatives (torch.Tensor): negative samples

        Returns:
            torch.Tensor: loss value
        """
        # calculate loss
        positive_distances = tree.calc_distance_between_subtree_weights(
            anchors, positives
        )
        negative_distances = tree.calc_distance_between_subtree_weights(
            anchors, negatives
        )
        return torch.mean(
            torch.max(
                positive_distances - negative_distances + self.margin,
                torch.zeros_like(positive_distances),
            )
        )


class NCELoss(nn.Module):
    """NCE loss function

    Attributes:
        temperature (float): Temperature
    """

    def __init__(self, temperature: float):
        """initialize NCE loss function

        Args:
            temperature (float): hyperparameter for NCE loss
        """
        super().__init__()
        self.temperature = temperature
        self.clamp_threshold = 1e-10

    def forward(
        self,
        tree: WeisfeilerLemanLabelingTree,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """calculate NCE loss

        Args:
            tree (WeisfeilerLemanLabelingTree): WLLT
            anchors (torch.Tensor): anchor samples
            positives (torch.Tensor): positive samples
            negatives (torch.Tensor): negative samples

        Returns:
            torch.Tensor: loss value
        """
        positive_distances = tree.calc_distance_between_subtree_weights(
            anchors, positives
        )
        negative_distances = tree.calc_distance_between_subtree_weights(
            anchors, negatives
        )
        positive_like = torch.exp(-positive_distances / self.temperature)
        negative_like = torch.exp(-negative_distances / self.temperature)
        return torch.mean(
            positive_distances / self.temperature
            + torch.log(
                torch.clamp(positive_like + negative_like, min=self.clamp_threshold)
            )
        )


class InfoNCELoss(nn.Module):
    """InfoNCE loss function

    Attributes:
        temperature (float): Temperature
    """

    def __init__(self, temperature: float):
        """initialize InfoNCE loss function

        Args:
            temperature (float): hyperparameter for NCE loss
        """
        super().__init__()
        self.temperature = temperature
        self.clamp_threshold = 1e-10

    def forward(
        self,
        tree: WeisfeilerLemanLabelingTree,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """calculate InfoNCE loss

        Args:
            tree (WeisfeilerLemanLabelingTree): WLLT
            anchors (torch.Tensor): anchor samples (B, D)
            positives (torch.Tensor): positive samples (B, D)
            negatives (torch.Tensor): negative samples (B, N, D)

        Returns:
            torch.Tensor: loss value
        """
        positive_distances = tree.calc_distance_between_subtree_weights(
            anchors, positives
        )  # (B,)
        negative_distances = tree.calc_distance_between_subtree_weights(
            anchors.unsqueeze(1).expand(negatives.shape), negatives
        )  # (B, N)
        positive_like = torch.exp(-positive_distances / self.temperature)
        negative_like = torch.exp(-negative_distances / self.temperature)
        return torch.mean(
            positive_distances / self.temperature
            + torch.log(
                torch.clamp(
                    positive_like + torch.sum(negative_like, dim=1),
                    min=self.clamp_threshold,
                )
            )
        )


class AllPairNCELoss(nn.Module):
    """NCE loss function

    Attributes:
        temperature (float): Temperature
    """

    def __init__(self, temperature: float, alpha: float):
        """initialize NCE loss function

        Args:
            temperature (float): hyperparameter for ALlPairNCE loss
            alpha (float): hyperparameter for AllPairNCE loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.clamp_threshold = 1e-10

    def forward(
        self,
        tree: WeisfeilerLemanLabelingTree,
        samples: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """calculate NCE loss

        Args:
            tree (WeisfeilerLemanLabelingTree): WLLT
            samples (torch.Tensor): X
            labels (torch.Tensor): Y

        Returns:
            torch.Tensor: loss value
        """
        n_samples = len(samples)
        distances = tree.calc_distance_between_subtree_weights(
            samples.repeat_interleave(n_samples, dim=0),
            samples.repeat(n_samples, 1),
        ).reshape(n_samples, n_samples)
        kernels = torch.exp(-distances / self.temperature)
        same_mask = torch.logical_and(
            labels.reshape(-1, 1) == labels.reshape(1, -1),
            torch.logical_not(torch.eye(n_samples, dtype=torch.bool)),
        )
        diff_mask = labels.reshape(-1, 1) != labels.reshape(1, -1)
        kernels_sum_same_label = torch.sum(kernels * same_mask, dim=1).reshape(-1, 1)
        kernels_sum_diff_label = torch.sum(kernels * diff_mask, dim=1).reshape(-1, 1)
        return torch.mean(
            same_mask
            * (
                distances / self.temperature
                + torch.log(
                    torch.clamp(
                        kernels_sum_same_label + self.alpha * kernels_sum_diff_label,
                        min=self.clamp_threshold,
                    )
                )
            )
        )


class KnnNCELoss(nn.Module):
    """NCE loss function

    Attributes:
        temperature (float): Temperature
    """

    def __init__(self, temperature: float, alpha: float, n_neighbors: int):
        """initialize NCE loss function

        Args:
            temperature (float): hyperparameter for ALlPairNCE loss
            alpha (float): hyperparameter for AllPairNCE loss
            n_neighbors (int): number of neighbors to consider
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.clamp_threshold = 1e-10

    def forward(
        self,
        tree: WeisfeilerLemanLabelingTree,
        samples: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """calculate NCE loss

        Args:
            tree (WeisfeilerLemanLabelingTree): WLLT
            samples (torch.Tensor): X
            labels (torch.Tensor): Y

        Returns:
            torch.Tensor: loss value
        """
        n_samples = len(samples)
        distances = tree.calc_distance_between_subtree_weights(
            samples.repeat_interleave(n_samples, dim=0),
            samples.repeat(n_samples, 1),
        ).reshape(n_samples, n_samples)
        kernels = torch.exp(-distances / self.temperature)

        same_mask = torch.logical_and(
            labels.reshape(-1, 1) == labels.reshape(1, -1),
            torch.logical_not(torch.eye(n_samples, dtype=torch.bool)),
        )
        diff_mask = labels.reshape(-1, 1) != labels.reshape(1, -1)
        kernels_diag_diff_minus = torch.where(
            torch.logical_or(diff_mask, torch.eye(n_samples, dtype=torch.bool)),
            -1,
            kernels,
        )
        _, kernels_topk_indices = torch.topk(
            kernels_diag_diff_minus, min(self.n_neighbors, n_samples), dim=1
        )
        same_neighbor_mask = torch.logical_and(
            same_mask,
            torch.zeros_like(same_mask, dtype=torch.bool).scatter_(
                1, kernels_topk_indices, True
            ),
        )

        kernels_sum_same_label_neighbor = torch.sum(
            kernels * same_neighbor_mask, dim=1
        ).reshape(-1, 1)
        kernels_sum_diff_label = torch.sum(kernels * diff_mask, dim=1).reshape(-1, 1)
        return torch.mean(
            same_neighbor_mask
            * (
                distances / self.temperature
                + torch.log(
                    torch.clamp(
                        kernels_sum_same_label_neighbor
                        + self.alpha * kernels_sum_diff_label,
                        min=self.clamp_threshold,
                    )
                )
            )
        )
