from typing import Optional

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import torch
import torch_geometric.utils as utils  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from torch_geometric.data import Data  # type: ignore

from tree import WeisfeilerLemanLabelingTree


def visualize_WLLT(
    tree: WeisfeilerLemanLabelingTree, path: str, withweight: bool = True
) -> None:
    """visualize WLLT

    Args:
        tree (WeisfeilerLemanLabelingTree): WLLT
        path (str): Path to save image
        withweight (bool, optional): Whether to show weight. Defaults to True.
    """
    nx_tree = nx.Graph()
    nx_tree.add_node(-1, label=-1)
    for node_idx in range(tree.n_nodes):
        nx_tree.add_node(node_idx, label=node_idx)
        nx_tree.add_edge(
            node_idx, tree.parent[node_idx], weight=tree.weight[node_idx].item()
        )
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(nx_tree)
    nx.draw(nx_tree, pos, with_labels=True)
    if withweight:
        nx.draw_networkx_edge_labels(
            nx_tree,
            pos,
            edge_labels=nx.get_edge_attributes(nx_tree, "weight"),
        )
    plt.savefig(path)
    plt.close()


def visualize_graph(
    graph: Data,
    path: str,
    node_dict: Optional[dict] = None,
    edge_dict: Optional[dict] = None,
) -> None:
    """visualize graph

    Args:
        graph (Data): graph
        path (str): Path to save image
        node_dict (Optional[dict]): Convert node attribute from integer. Defaults to None.
        edge_dict (Optional[dict]): Convert edge attribute from integer. Defaults to None.
    """
    nx_graph = nx.Graph()
    int2elem = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
    for node_idx, node_attr in enumerate(torch.argmax(graph.x, dim=1)):
        if node_dict is not None:
            nx_graph.add_node(node_idx, label=node_dict[node_attr.item()])
        else:
            nx_graph.add_node(node_idx, label=node_attr.item())
    for (u, v), edge_attr in zip(
        graph.edge_index.T, torch.argmax(graph.edge_attr, dim=1)
    ):
        if edge_dict is not None:
            nx_graph.add_edge(u.item(), v.item(), label=edge_dict[edge_attr.item()])
        else:
            nx_graph.add_edge(u.item(), v.item(), label=edge_attr.item())
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos)
    nx.draw_networkx_labels(
        nx_graph, pos, labels=nx.get_node_attributes(nx_graph, "label")
    )
    nx.draw_networkx_edge_labels(
        nx_graph,
        pos,
        edge_labels=nx.get_edge_attributes(nx_graph, "label"),
    )
    plt.savefig(path)
    plt.close()
