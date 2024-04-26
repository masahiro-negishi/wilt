import torch
from torch_geometric.data import Batch, Data, Dataset  # type: ignore


class WeisfeilerLemanLabelingTree:
    """Weisfeiler Leman Labeling Tree

    Attributes:
        depth (int): Number of layers in WLLT
        exp_parameter (bool): Whether to set weight as exp(parameter)
        norm (bool): Whether to normalize the distribution
        n_nodes (int): Number of nodes in WLLT
        parent (list[int]): Parent node of each node in WLLT
        attr2label (dict[int, int]): Mapping from attribute to label (for layer 0)
        labeling_hash (dict[tuple[int, tuple[int, ...]], int]): Hash table of labelings (for layer 1, layer 2, ..., layer depth)
        parameter (torch.Tensor): Parameter of each edge in WLLT

    Properties:
        weight (torch.Tensor): Weight of each edge in WLLT

    Structure:
        __init__
        |-- _build_tree
            |-- _adjancy_list
            |-- reset_parameter

        weight

        calc_distance_between_graphs
        |-- calc_subtree_weights
        |-- calc_distance_between_subtree_weights

        test_data_wwo_unseen_nodes

        reset_parameter
        load_parameter
        train
        eval
    """

    ##################
    # initialization #
    ##################
    def __init__(
        self, data: Dataset, depth: int, exp_parameter: bool = True, norm: bool = False
    ) -> None:
        """initialize WLLT

        Args:
            data (Dataset): Dataset
            depth (int): Number of layers in WLLT
            exp_parameter (bool, optional): Whether to set weight as exp(parameter). Defaults to True.
            norm (bool, optional): Whether to normalize the distribution. Defaults to False.
        """
        self.depth = depth
        self.exp_parameter = exp_parameter
        self.norm = norm
        assert self.depth > 0, "Depth should be greater than 0"
        self._build_tree(data)

    def _adjancy_list(self, graph: Data) -> list[list[int]]:
        """convert edge_index to adjancy list

        Args:
            graph (Data): graph

        Returns:
            list[list[int]]: adjancy list
        """
        adj_list: list[list[int]] = [[] for _ in range(graph.num_nodes)]
        for u, v in graph.edge_index.T:
            adj_list[u.item()].append(v.item())
        return adj_list

    def _build_tree(self, data) -> None:
        """Build WLLT tree

        Args:
            data (Dataset): Dataset
        """
        cnt_nodes: list[int] = [
            0 for _ in range(self.depth + 1)
        ]  # cnt_nodes[i] = number of nodes in layer [i] (The layer of root node is 0)
        parent: list[list[int]] = [
            [] for _ in range(self.depth + 1)
        ]  # parent[i][j] = parent node of node j in layer i (parent of root node is -1)
        attr2label: dict[int, int] = {}  # attr2label[i] = label of attribute i
        labeling_hash: list[dict[tuple[int, tuple[int, ...]], int]] = [
            {} for _ in range(self.depth + 1)
        ]  # labeling_hash[i] = dict of labelings in layer i (labeling_hash[0] is not used)

        # root node
        cnt_nodes[0] = 1
        parent[0].append(-1)

        # iterate over dataset # O(|G| * (|E| + self.depth * |V| * log|E|))
        for g in data:
            # initial labeling # O(|V|)
            current_labeling: list[int] = [-1 for _ in range(g.num_nodes)]
            for node_idx, node_attr in enumerate(torch.argmax(g.x, dim=1)):
                if attr2label.get(node_attr.item()) is None:
                    attr2label[node_attr.item()] = cnt_nodes[1]
                    cnt_nodes[1] += 1
                    parent[1].append(0)  # root node
                current_labeling[node_idx] = attr2label[node_attr.item()]
            # adjancy_list # O(|V| + |E|)
            adj_list: list[list[int]] = self._adjancy_list(g)
            # iterative labeling # O(self.depth * |V| * log|E|)
            for d in range(2, self.depth + 1):
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
        self.attr2label: dict[int, int] = {
            k: tuple2int[(1, v)] for k, v in attr2label.items()
        }
        self.labeling_hash: dict[tuple[int, tuple[int, ...]], int] = {}
        for layer_idx in range(2, self.depth + 1):
            for key, val in labeling_hash[layer_idx].items():
                label_idx = tuple2int[(layer_idx - 1, key[0])]
                neighbors_idx = tuple(
                    [tuple2int[(layer_idx - 1, nx)] for nx in key[1]]
                )  # already sorted
                self.labeling_hash[(label_idx, neighbors_idx)] = tuple2int[
                    (layer_idx, val)
                ]

        # weight
        self.reset_parameter()

    ########################
    # distance calculation #
    ########################
    def calc_subtree_weights(self, graph: Data) -> torch.Tensor:
        """calculate distribution on WLLT

        Args:
            graph (Data): graph

        Returns:
            torch.Tensor: distribution on WLLT
        """
        dist = torch.zeros(self.n_nodes, dtype=torch.float32)
        current_labeling: list[int] = [-1 for _ in range(graph.num_nodes)]
        # root node
        dist[0] = graph.num_nodes
        # initial labeling
        for node_idx, node_attr in enumerate(torch.argmax(graph.x, dim=1)):
            current_labeling[node_idx] = self.attr2label[node_attr.item()]
            dist[self.attr2label[node_attr.item()]] += 1
        # adjancy_list
        adj_list: list[list[int]] = self._adjancy_list(graph)
        # iterative labeling
        for _ in range(2, self.depth + 1):
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
        if self.norm:
            dist /= graph.num_nodes
        return dist

    def calc_distance_between_subtree_weights(
        self,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
    ) -> torch.Tensor:
        """calculate distance between two distributions on WLLT

        Args:
            weight1 (torch.Tensor): subtree_weight vector(s)
            weight2 (torch.Tensor): subtree_weight vector(s)

        Returns:
            torch.Tensor: distance(s)
        """
        if len(weight1.shape) == 1:
            return torch.dot(
                torch.abs(weight1 - weight2),
                self.weight,
            )  # (1, )
        else:
            # batch execution
            return torch.abs(weight1 - weight2) @ self.weight  # (batch_size,)

    def calc_distance_between_graphs(
        self,
        graph1: Data | Batch | list[Data],
        graph2: Data | Batch | list[Data],
    ) -> torch.Tensor:
        """calculate distance between two graphs

        Args:
            graph1 (Data | Batch): graph(s)
            graph2 (Data | Batch): graph(s)

        Returns:
            torch.Tensor: distance(s)
        """
        if isinstance(graph1, Data) and isinstance(graph2, Data):
            graph1 = [graph1]
            graph2 = [graph2]
        elif isinstance(graph1, Batch) and isinstance(graph2, Batch):
            graph1 = graph1.to_data_list()
            graph2 = graph2.to_data_list()
        dist1 = torch.stack(
            [self.calc_subtree_weights(g) for g in graph1], dim=0
        )  # (batch_size, self.n_nodes)
        dist2 = torch.stack(
            [self.calc_subtree_weights(g) for g in graph2], dim=0
        )  # (batch_size, self.n_nodes)
        return self.calc_distance_between_subtree_weights(dist1, dist2)

    ########################
    # test data separation #
    ########################
    def test_data_wwo_unseen_nodes(
        self, data: Dataset, train_indices: torch.Tensor, test_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """classify test data into ones with and without unseen nodes

        Args:
            data (Dataset): Dataset
            train_indices (torch.Tensor): indices of training data
            test_indices (torch.Tensor): indices of test data

        Returns:
            tuple[torch.Tensor, torch.Tensor]: indices for test data with/without unseen nodes
        """
        train_data = data[train_indices]
        test_data = data[test_indices]
        train_seen = torch.zeros(self.n_nodes, dtype=torch.bool)
        for graph in train_data:
            train_dist = self.calc_subtree_weights(graph)
            train_seen = torch.logical_or(train_seen, train_dist > 0)
        train_unseen = torch.logical_not(train_seen)
        test_seen_indices = []
        test_unseen_indices = []
        for i, graph in zip(test_indices, test_data):
            test_dist = self.calc_subtree_weights(graph)
            if torch.any(torch.logical_and(test_dist > 0, train_unseen)):
                test_unseen_indices.append(i)
            else:
                test_seen_indices.append(i)
        return (
            torch.tensor(test_seen_indices),
            torch.tensor(test_unseen_indices),
        )

    ##############
    # parameters #
    ##############
    @property
    def weight(self) -> torch.Tensor:
        if self.exp_parameter:
            return torch.exp(self.parameter)
        else:
            return self.parameter

    def reset_parameter(self) -> None:
        if self.exp_parameter:
            self.parameter: torch.Tensor = torch.zeros(
                self.n_nodes, dtype=torch.float32
            )
        else:
            self.parameter = torch.ones(self.n_nodes, dtype=torch.float32)

    def load_parameter(self, path: str) -> None:
        self.parameter = torch.load(path)

    def train(self) -> None:
        self.parameter.requires_grad = True

    def eval(self) -> None:
        self.parameter.requires_grad = False
