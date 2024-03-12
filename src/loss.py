import random

import torch
from torch import nn

from tree import WeisfeilerLemanLabelingTree
import numpy as np


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
        self.float32_max = np.finfo(np.float32).max
        self.float32_smallest_positive = np.finfo(np.float32).smallest_normal

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
        # calculate loss
        positive_distances = tree.calc_distance_between_subtree_weights(
            anchors, positives
        )
        negative_distances = tree.calc_distance_between_subtree_weights(
            anchors, negatives
        )
        max_distances = torch.max(positive_distances, negative_distances)
        positive_like = torch.exp(
            (positive_distances - max_distances) / self.temperature
        )
        negative_like = torch.exp(
            (negative_distances - max_distances) / self.temperature
        )
        return torch.mean(
            -torch.log(
                torch.clip(
                    positive_like / (positive_like + negative_like),
                    min=self.float32_smallest_positive,
                )
            )
        )
