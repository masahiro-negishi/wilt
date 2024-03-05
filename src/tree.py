import os

import torch
import torch_geometric.data  # type: ignore
from torch_geometric.datasets import TUDataset  # type: ignore

from path import DATA_DIR


class WeisfeilerLemanLabelingTree:
    """Weisfeiler Leman Labeling Tree

    Attributes:
        dataset_name (str): Name of dataset
        dataset_root (str): Root directory of dataset
        data (TUDataset): Dataset
        depth (int): Number of layers in WWLLT
        n_nodes (int): Number of nodes in WWLLT
        parent (list[int]): Parent node of each node in WWLLT
        attr2label (dict[int, int]): Mapping from attribute to label (for layer 0)
        labeling_hash (dict[tuple[int, tuple[int, ...]], int]): Hash table of labelings (for layer 1 to layer depth)
        weight (torch.Tensor): Weight of each node in WWLLT

    Structure:
        __init__
        |-- _build_tree
            |-- _adjancy_list

        calc_distance
        |-- _calc_dist_on_tree
        |   |-- _adjancy_list
        |-- _calc_subtree_weight
    """

    def __init__(self, dataset_name: str, depth: int) -> None:
        """initialize WWLLT

        Args:
            dataset_name (str): e.g. "MUTAG"
            depth (int): Number of layers in WWLLT
        """
        self.dataset_name = dataset_name
        self.dataset_root = os.path.join(DATA_DIR, "TUDataset")
        self.data = TUDataset(root=self.dataset_root, name=self.dataset_name)
        self.depth = depth
        assert self.depth > 0, "Depth should be greater than 0"
        self._build_tree()

    def _adjancy_list(self, graph: torch_geometric.data.Data) -> list[list[int]]:
        """convert edge_index to adjancy list

        Args:
            graph (torch_geometric.data.Data): graph

        Returns:
            list[list[int]]: adjancy list
        """
        adj_list: list[list[int]] = [[] for _ in range(graph.num_nodes)]
        for u, v in graph.edge_index.T:
            adj_list[u].append(v)
        return adj_list

    def _build_tree(self) -> None:
        """Build WWLLT tree"""
        cnt_nodes: list[int] = [
            0 for _ in range(self.depth + 1)
        ]  # cnt_nodes[i] = number of nodes in layer [i] (The layer of root node is -1)
        parent: list[list[int]] = [
            [] for _ in range(self.depth + 1)
        ]  # parent[i][j] = parent node of node j in layer i
        attr2label: dict[int, int] = {}  # attr2label[i] = label of attribute i
        labeling_hash: list[dict[tuple[int, tuple[int, ...]], int]] = [
            {} for _ in range(self.depth + 1)
        ]  # labeling_hash[i] = dict of labelings in layer i (labeling_hash[0] is not used)

        # iterate over dataset # O(|G| * (|E| + self.depth * |V| * log|E|))
        for g in self.data:
            # initial labeling # O(|V|)
            current_labeling: list[int] = [-1 for _ in range(g.num_nodes)]
            for node_idx, node_attr in enumerate(torch.argmax(g.x, dim=1)):
                if attr2label.get(node_attr.item()) is None:
                    attr2label[node_attr.item()] = cnt_nodes[0]
                    cnt_nodes[0] += 1
                    parent[0].append(-1)  # root node
                current_labeling[node_idx] = attr2label[node_attr.item()]
            # adjancy_list # O(|V| + |E|)
            adj_list: list[list[int]] = self._adjancy_list(g)
            # iterative labeling # O(self.depth * |V| * log|E|)
            for d in range(1, self.depth + 1):
                new_labeling: list[int] = [-1 for _ in range(g.num_nodes)]
                for node_idx in range(g.num_nodes):
                    # TODO: linear time sort
                    idx: tuple[int, tuple[int, ...]] = (
                        current_labeling[node_idx],
                        tuple(
                            sorted([current_labeling[nx] for nx in adj_list[node_idx]])
                        ),
                    )
                    if labeling_hash[d].get(idx) is None:
                        labeling_hash[d][idx] = cnt_nodes[d]
                        cnt_nodes[d] += 1
                        parent[d].append(current_labeling[node_idx])
                    new_labeling[node_idx] = labeling_hash[d][idx]
                current_labeling = new_labeling

        # relabeling
        tuple2int: dict[tuple[int, int], int] = {}
        n_nodes = 0
        for layer_idx in range(self.depth + 1):
            for label_idx in range(cnt_nodes[layer_idx]):
                tuple2int[(layer_idx, label_idx)] = n_nodes
                n_nodes += 1
        self.n_nodes = n_nodes  # sum(cnt_nodes)
        self.parent: list[int] = [-1 for _ in range(self.n_nodes)]
        for layer_idx in range(1, self.depth + 1):
            for label_idx in range(cnt_nodes[layer_idx]):
                self.parent[tuple2int[(layer_idx, label_idx)]] = tuple2int[
                    (layer_idx - 1, parent[layer_idx][label_idx])
                ]
        self.attr2label: dict[int, int] = attr2label
        self.labeling_hash: dict[tuple[int, tuple[int, ...]], int] = {}
        for layer_idx in range(1, self.depth + 1):
            for key, val in labeling_hash[layer_idx].items():
                label_idx = tuple2int[(layer_idx - 1, key[0])]
                neighbors_idx = tuple([tuple2int[(layer_idx - 1, nx)] for nx in key[1]])
                self.labeling_hash[(label_idx, neighbors_idx)] = tuple2int[
                    (layer_idx, val)
                ]

        # weight
        self.weight: torch.Tensor = torch.ones(self.n_nodes, dtype=torch.float32)

    def _calc_dist_on_tree(self, graph: torch_geometric.data.Data) -> torch.Tensor:
        """calculate distribution on WWLLT

        Args:
            graph (torch_geometric.data.Data): graph

        Returns:
            torch.Tensor: distribution on WWLLT
        """
        dist = torch.zeros(self.n_nodes, dtype=torch.float32)
        current_labeling: list[int] = [-1 for _ in range(graph.num_nodes)]
        # initial labeling
        for node_idx, node_attr in enumerate(torch.argmax(graph.x, dim=1)):
            current_labeling[node_idx] = self.attr2label[node_attr.item()]
            dist[self.attr2label[node_attr.item()]] += 1
        # adjancy_list
        adj_list: list[list[int]] = self._adjancy_list(graph)
        # iterative labeling
        for d in range(1, self.depth + 1):
            new_labeling: list[int] = [-1 for _ in range(graph.num_nodes)]
            for node_idx in range(graph.num_nodes):
                # TODO: linear time sort
                idx: tuple[int, tuple[int, ...]] = (
                    current_labeling[node_idx],
                    tuple(sorted([current_labeling[nx] for nx in adj_list[node_idx]])),
                )
                new_labeling[node_idx] = self.labeling_hash[idx]
                dist[self.labeling_hash[idx]] += 1
            current_labeling = new_labeling
        # normalize
        # TODO: consider dist /= dist.sum()
        dist /= graph.num_nodes
        return dist

    def _calc_subtree_weight(self, dist: torch.Tensor) -> torch.Tensor:
        """calculate weight of each subtree in WWLLT

        Args:
            dist (torch.Tensor): distribution on WWLLT

        Returns:
            torch.Tensor: weight of each subtree in WWLLT
        """
        weight = dist.clone()
        for node_idx in range(self.n_nodes - 1, -1, -1):
            weight[self.parent[node_idx]] += weight[node_idx]
        return weight

    def calc_distance(
        self,
        graph1: list[torch_geometric.data.Data],
        graph2: list[torch_geometric.data.Data],
    ) -> torch.Tensor:
        dist_1 = [
            self._calc_dist_on_tree(g) for g in graph1
        ]  # (batch_size, self.n_nodes)
        dist_2 = [
            self._calc_dist_on_tree(g) for g in graph2
        ]  # (batch_size, self.n_nodes)
        weight_1 = torch.stack(
            [self._calc_subtree_weight(d) for d in dist_1], dim=0
        )  # (batch_size, self.n_nodes)
        weight_2 = torch.stack(
            [self._calc_subtree_weight(d) for d in dist_2], dim=0
        )  # (batch_size, self.n_nodes)
        return torch.abs(weight_1 - weight_2) @ self.weight  # (batch_size,)


if __name__ == "__main__":
    wwllt = WeisfeilerLemanLabelingTree(dataset_name="MUTAG", depth=3)
