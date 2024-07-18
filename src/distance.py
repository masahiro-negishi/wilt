import argparse
import copy
import os

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import numpy as np
import ot  # type: ignore
import timeout_decorator  # type: ignore
import torch
from torch_geometric.data import Data, Dataset  # type: ignore
from torch_geometric.datasets import TUDataset  # type: ignore
from torch_geometric.utils import to_networkx  # type: ignore

from path import DATA_DIR, DIS_MX_DIR, GNN_DIR
from tree import WeisfeilerLemanLabelingTree

NUM_PAIR = 1000
FOLD = 0
KFOLD = 5


def calc_distance_matrix_WLLT_WWLGK(dataset: Dataset, metric, **kwargs) -> torch.Tensor:
    tree = WeisfeilerLemanLabelingTree(
        dataset,
        kwargs["depth"],
        False,
        kwargs["normalize"] if metric == "WLLT" else True,
        False if metric == "WWLGK" else None,
    )
    indices = np.random.RandomState(seed=0).permutation(len(dataset) * len(dataset))[
        :NUM_PAIR
    ]
    embeddings = torch.stack([tree.calc_subtree_weights(g) for g in dataset], dim=0)
    tree.eval()
    distance_matrix = torch.zeros(NUM_PAIR)
    for i in range(NUM_PAIR):
        distance_matrix[i] = tree.calc_distance_between_subtree_weights(
            embeddings[indices[i] // len(dataset)],
            embeddings[indices[i] % len(dataset)],
        )
    if metric == "WWLGK":
        distance_matrix /= 2 * kwargs["depth"]
    return distance_matrix


def calc_distance_matrix_GED(dataset: Dataset, **kwargs) -> torch.Tensor:
    ANS = -1

    @timeout_decorator.timeout(kwargs["timeout"])
    def call_ged(g1, g2):
        nonlocal ANS
        for tmpd in nx.optimize_graph_edit_distance(to_networkx(g1), to_networkx(g2)):
            ANS = tmpd

    distance_matrix = torch.zeros(NUM_PAIR)
    indices = np.random.RandomState(seed=0).permutation(len(dataset) * len(dataset))[
        :NUM_PAIR
    ]
    for i in range(NUM_PAIR):
        g1 = dataset[indices[i] // len(dataset)]
        g2 = dataset[indices[i] % len(dataset)]
        ANS = -1
        try:
            call_ged(g1, g2)
        except:
            pass
        if ANS == -1:
            raise ValueError
        distance_matrix[i] = ANS
        print(i, ANS)
    return distance_matrix


def get_neighbors(g: Data) -> dict[int, list[int]]:
    """neighbor indexes for each node

    Args:
        g (Data): input torch_geometric graph

    Returns:
        dict[int, :list[int]]: a dictionary that store the neighbor indexes

    Notes:
        Copied from https://github.com/chingyaoc/TMD
    """
    adj: dict[int, list[int]] = {}
    for i in range(len(g.edge_index[0])):
        node1 = g.edge_index[0][i].item()
        node2 = g.edge_index[1][i].item()
        if node1 in adj.keys():
            adj[node1].append(node2)
        else:
            adj[node1] = [node2]
    return adj


def TMD(g1: Data, g2: Data, w: float | list[float], L: int = 4):
    """Tree Mover's Distance between two graphs.

    Args:
        g1 (Data): graph 1
        g2 (Data): graph 2
        w (float | list[float]): weight constant(s) for each depth
        L (int, optional): Depth of computation trees for calculating TMD. Defaults to 4.

    Returns:
        float: The TMD between g1 and g2

    Notes:
        Copied from https://github.com/chingyaoc/TMD
    """
    if isinstance(w, list):
        assert len(w) == L - 1
    else:
        w = [w] * (L - 1)

    # get attributes
    n1, n2 = len(g1.x), len(g2.x)
    feat1, feat2 = g1.x, g2.x
    adj1 = get_neighbors(g1)
    adj2 = get_neighbors(g2)

    D = np.zeros((n1, n2))

    # level 1 (pair wise distance)
    M = np.zeros((n1 + 1, n2 + 1))
    for i in range(n1):
        for j in range(n2):
            D[i, j] = torch.norm(feat1[i] - feat2[j])
            M[i, j] = D[i, j]
    # distance w.r.t. blank node
    M[:n1, n2] = torch.norm(feat1, dim=1)
    M[n1, :n2] = torch.norm(feat2, dim=1)

    # level l (tree OT)
    for l in range(L - 1):
        M1 = copy.deepcopy(M)
        M = np.zeros((n1 + 1, n2 + 1))

        # calculate pairwise cost between tree i and tree j
        for i in range(n1):
            for j in range(n2):
                try:
                    degree_i = len(adj1[i])
                except:
                    degree_i = 0
                try:
                    degree_j = len(adj2[j])
                except:
                    degree_j = 0

                if degree_i == 0 and degree_j == 0:
                    M[i, j] = D[i, j]
                # if degree of node is zero, calculate TD w.r.t. blank node
                elif degree_i == 0:
                    wass = 0.0
                    for jj in range(degree_j):
                        wass += M1[n1, adj2[j][jj]]
                    M[i, j] = D[i, j] + w[l] * wass
                elif degree_j == 0:
                    wass = 0.0
                    for ii in range(degree_i):
                        wass += M1[adj1[i][ii], n2]
                    M[i, j] = D[i, j] + w[l] * wass
                # otherwise, calculate the tree distance
                else:
                    max_degree = max(degree_i, degree_j)
                    if degree_i < max_degree:
                        cost = np.zeros((degree_i + 1, degree_j))
                        cost[degree_i] = M1[n1, adj2[j]]
                        dist_1, dist_2 = np.ones(degree_i + 1), np.ones(degree_j)
                        dist_1[degree_i] = max_degree - float(degree_i)
                    else:
                        cost = np.zeros((degree_i, degree_j + 1))
                        cost[:, degree_j] = M1[adj1[i], n2]
                        dist_1, dist_2 = np.ones(degree_i), np.ones(degree_j + 1)
                        dist_2[degree_j] = max_degree - float(degree_j)
                    for ii in range(degree_i):
                        for jj in range(degree_j):
                            cost[ii, jj] = M1[adj1[i][ii], adj2[j][jj]]
                    wass = ot.emd2(dist_1, dist_2, cost)

                    # summarize TMD at level l
                    M[i, j] = D[i, j] + w[l] * wass

        # fill in dist w.r.t. blank node
        for i in range(n1):
            try:
                degree_i = len(adj1[i])
            except:
                degree_i = 0

            if degree_i == 0:
                M[i, n2] = torch.norm(feat1[i])
            else:
                wass = 0.0
                for ii in range(degree_i):
                    wass += M1[adj1[i][ii], n2]
                M[i, n2] = torch.norm(feat1[i]) + w[l] * wass

        for j in range(n2):
            try:
                degree_j = len(adj2[j])
            except:
                degree_j = 0
            if degree_j == 0:
                M[n1, j] = torch.norm(feat2[j])
            else:
                wass = 0.0
                for jj in range(degree_j):
                    wass += M1[n1, adj2[j][jj]]
                M[n1, j] = torch.norm(feat2[j]) + w[l] * wass

    # final OT cost
    max_n = max(n1, n2)
    dist_1, dist_2 = np.ones(n1 + 1), np.ones(n2 + 1)
    if n1 < max_n:
        dist_1[n1] = max_n - float(n1)
        dist_2[n2] = 0.0
    else:
        dist_1[n1] = 0.0
        dist_2[n2] = max_n - float(n2)

    wass = ot.emd2(dist_1, dist_2, M)
    return wass


def calc_distance_matrix_TMD(dataset: Dataset, **kwargs) -> torch.Tensor:
    distance_matrix = torch.zeros(NUM_PAIR)
    indices = np.random.RandomState(seed=0).permutation(len(dataset) * len(dataset))[
        :NUM_PAIR
    ]
    ws = [
        None,
        [1 / 1],
        [1 / 2, 2 / 1],
        [1 / 3, 3 / 3, 3 / 1],
        [1 / 4, 4 / 6, 6 / 4, 4 / 1],
    ]
    for i in range(NUM_PAIR):
        g1 = dataset[indices[i] // len(dataset)]
        g2 = dataset[indices[i] % len(dataset)]
        distance_matrix[i] = TMD(g1, g2, ws[kwargs["depth"] - 1], kwargs["depth"])
    return distance_matrix


def calc_distance_matrix(
    dataset_name: str,
    metric: str,
    path: str,
    **kwargs,
) -> None:
    """calculate the distance matrix of the dataset using the specified metric, and save the distance matrix in the specified path.

    Args:
        dataset_name (str): The name of the dataset.
        metric (str): The name of the metric.
        path (str): The path to save the distance matrix.
        **kwargs: additional arguments
    """
    dataset = TUDataset(root=os.path.join(DATA_DIR, "TUDataset"), name=dataset_name)
    if metric == "WLLT" or metric == "WWLGK":
        distance_matrix = calc_distance_matrix_WLLT_WWLGK(dataset, metric, **kwargs)
    elif metric == "GED":
        distance_matrix = calc_distance_matrix_GED(dataset, **kwargs)
    elif metric == "TMD":
        distance_matrix = calc_distance_matrix_TMD(dataset, **kwargs)
    else:
        raise NotImplementedError
    torch.save(distance_matrix, path)


def compare_distance_matrix(
    dataset_name: str,
    dis_mx_mpnn: torch.Tensor,
    path: str,
) -> None:
    """_summary_

    Args:
        dataset_name (str): name of the dataset
        dis_mx_mpnn (torch.Tensor): distance matrix from GNN embeddings
        path (str): path to save the scatter plot
    """
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    indices = np.random.RandomState(seed=0).permutation(
        len(dis_mx_mpnn) * len(dis_mx_mpnn)
    )[:NUM_PAIR]
    for i, ylabel in enumerate(["WLLT (unnorm)", "WLLT (norm)", "WWLGK", "TMD", "GED"]):
        for j in range(4):
            if i == 0:
                path_dis_mx = os.path.join(
                    DIS_MX_DIR,
                    dataset_name,
                    f"WLLT_d={j+1}_norm=False.pt",
                )
            elif i == 1:
                path_dis_mx = os.path.join(
                    DIS_MX_DIR,
                    dataset_name,
                    f"WLLT_d={j+1}_norm=True.pt",
                )
            elif i == 2:
                path_dis_mx = os.path.join(
                    DIS_MX_DIR,
                    dataset_name,
                    f"WWLGK_d={j+1}.pt",
                )
            elif i == 3:
                path_dis_mx = os.path.join(
                    DIS_MX_DIR,
                    dataset_name,
                    f"TMD_d={j+1}.pt",
                )
            elif i == 4:
                path_dis_mx = os.path.join(
                    DIS_MX_DIR,
                    dataset_name,
                    f"GED_t=30.pt",
                )
                if j != 0:
                    continue
            dis_mx = torch.load(path_dis_mx)
            if i <= 2:
                axes[i, j].scatter(
                    dis_mx.flatten()[indices], dis_mx_mpnn.flatten()[indices]
                )
            else:
                axes[i, j].scatter(dis_mx.flatten(), dis_mx_mpnn.flatten()[indices])
            if i <= 3:
                axes[i, j].set_title(f"{ylabel} (d={j+1})")
            else:
                axes[i, j].set_title(f"{ylabel}")
            print(i, j)
    fig.supylabel("Distance between MPNN embeddings", size="xx-large")
    fig.supxlabel("Distance based on various metrics", size="xx-large")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", choices=["MUTAG", "Mutagenicity", "NCI1"])
    subparsers = parser.add_subparsers(dest="function")
    calc_parser = subparsers.add_parser("calc")
    calc_parser.add_argument("--metric", choices=["WLLT", "WWLGK", "GED", "TMD"])
    calc_parser.add_argument("--depth", type=int, required=False)
    calc_parser.add_argument("--normalize", action="store_true", required=False)
    calc_parser.add_argument("--timeout", type=int, required=False)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--gnn_depth", type=int)
    args = parser.parse_args()
    if args.function == "calc":
        if args.metric == "WLLT":
            path = os.path.join(
                DIS_MX_DIR,
                args.dataset_name,
                f"{args.metric}_d={args.depth}_norm={args.normalize}.pt",
            )
        elif args.metric == "WWLGK":
            path = os.path.join(
                DIS_MX_DIR,
                args.dataset_name,
                f"{args.metric}_d={args.depth}.pt",
            )
        elif args.metric == "GED":
            path = os.path.join(
                DIS_MX_DIR,
                args.dataset_name,
                f"{args.metric}_t={args.timeout}.pt",
            )
        elif args.metric == "TMD":
            path = os.path.join(
                DIS_MX_DIR,
                args.dataset_name,
                f"{args.metric}_d={args.depth}.pt",
            )
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        calc_distance_matrix(**args.__dict__, path=path)
    elif args.function == "compare":
        for model in ["gat", "gcn", "gin"]:
            for metric in ["l1", "l2"]:
                compare_distance_matrix(
                    args.dataset_name,
                    torch.load(
                        os.path.join(
                            GNN_DIR,
                            args.dataset_name,
                            model,
                            str(args.gnn_depth),
                            "fold0",
                            f"dist_{metric}.pt",
                        )
                    ),
                    os.path.join(
                        DIS_MX_DIR,
                        args.dataset_name,
                        f"{model}_d=3_{metric}.png",
                    ),
                )
