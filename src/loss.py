import random

import torch
from torch import nn

from tree import WeisfeilerLemanLabelingTree


class TripletLoss(nn.Module):
    """Triple loss function

    Attributes:
        margin (float): Margin
        n_classes (int): Number of classes
    """

    def __init__(self, margin: float, n_classes: int):
        """initialize triplet loss function

        Args:
            margin (float): hyperparameter for triplet loss
            n_classes (int): Number of classes
        """
        super().__init__()
        self.margin = margin
        self.n_classes = n_classes

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
