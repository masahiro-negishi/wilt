import json
import os

import matplotlib.pyplot as plt  # type: ignore
import torch  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.datasets import ZINC, MoleculeNet, TUDataset  # type: ignore

from path import DATA_DIR, GNN_DIR, RESULT_DIR
from tree import WeisfeilerLemanLabelingTree
from utils import load_dataset

model = "gcn"
embedding_dist = "l2"
loss = "l2"
coeff = 0.0001
pooling = "mean"
norm = "norm"
seed = 1

rslt_path = os.path.join(
    RESULT_DIR,
    "Mutagenicity",
    f"{model}",
    f"l=3_p={pooling}_d=64_s={seed}",
    embedding_dist,
    "d4",
    f"{norm}_l={loss}_a=True_l1={coeff}_b=256_e=10_lr=0.01_c=0.0_s=0_e=tree",
)

with open(os.path.join(rslt_path, "info.json")) as f:
    info = json.load(f)

data = load_dataset(info["dataset_name"])
print(data[278])
print(data[339])
print(data[1233])

tree = WeisfeilerLemanLabelingTree(data, int(info["depth"]), False, False)
tree.load_parameter(os.path.join(rslt_path, "fold0", "model_final.pt"))

sorted, indices = torch.sort(tree.weight, descending=True)
embeddings = torch.stack([tree.calc_subtree_weights(g) for g in data], dim=0)

if info["dataset_name"] == "synthetic_bin" or info["dataset_name"] == "Mutagenicity":
    n_positive = torch.sum(torch.tensor([g.y.item() for g in data]) == 1)
    n_negative = torch.sum(torch.tensor([g.y.item() for g in data]) == 0)
    predictions = torch.load(
        os.path.join(
            GNN_DIR,
            info["dataset_name"],
            model,
            f"l=3_p={pooling}_d=64_s={seed}",
            "fold0",
            "predictions.pt",
        )
    ).flatten()
    n_predicted_positive = torch.sum(predictions == 1)
    n_predicted_negative = torch.sum(predictions == 0)
    print(n_positive, n_negative, n_predicted_positive, n_predicted_negative)
    labels = torch.tensor([g.y.item() for g in data])

    os.makedirs(model, exist_ok=True)
    cnt = 0
    for idx, c in enumerate(indices):
        # if torch.sum(embeddings[:, c] > 0) < 10:
        if torch.sum(embeddings[:, c] > 0) < len(data) * 0.01:
            continue
        print(
            f"{c.item()}: w={sorted[idx].item()}\
            r:{idx+1}\
            p:{torch.sum(embeddings[labels==1][:,c]>0)}\
            n:{torch.sum(embeddings[labels==0][:,c]>0)}\
            pred_p:{torch.sum(embeddings[predictions==1][:,c]>0)}\
            pred_n:{torch.sum(embeddings[predictions==0][:,c]>0)}"
        )
        os.makedirs(f"{model}/{c}", exist_ok=True)
        tree.unfolding_tree(c.item(), f"{model}/{c}/tree.png")
        g_cnt = 0
        for i in range(len(data)):
            if torch.sum(embeddings[i]) > 30 * 4:
                continue
            if embeddings[i, c] > 0:
                os.makedirs(f"{model}/{c}/{i}_y={data[i].y}", exist_ok=True)
                tree.color_unfolding_tree_in_graph(
                    c, data[i], f"{model}/{c}/{i}_y={data[i].y}"
                )
                g_cnt += 1
            if g_cnt >= 3:
                break
        cnt += 1
        if cnt >= 20:
            break

    print(sorted[:100])

    plt.hist(tree.weight.detach().numpy(), bins=100)
    plt.savefig(f"{model}/weight.png")
    print(
        f"non zero: {torch.sum(tree.weight > 0)}, zero: {torch.sum(tree.weight == 0)}"
    )
elif info["dataset_name"] == "synthetic_mul" or info["dataset_name"] == "ENZYMES":
    classes = torch.unique(torch.tensor([g.y.item() for g in data]))
    n_classes = torch.tensor(
        [torch.sum(torch.tensor([g.y.item() for g in data]) == c) for c in classes]
    )
    predictions = torch.load(
        os.path.join(
            GNN_DIR, info["dataset_name"], model, "3", "fold0", "predictions.pt"
        )
    ).flatten()
    n_predicted = torch.tensor([torch.sum(predictions == c) for c in classes])
    print(n_classes, n_predicted)
    labels = torch.tensor([g.y.item() for g in data])

    os.makedirs(model, exist_ok=True)
    cnt = 0
    for idx, c in enumerate(indices):
        if torch.sum(embeddings[:, c] > 0) < 10:
            continue
        print(f"{c.item()}: w={sorted[idx].item()} r:{idx+1}")
        print(
            "label: ",
            [
                torch.sum(embeddings[labels == i][:, c] > 0).item()
                for i in range(len(classes))
            ],
        )
        print(
            "pred: ",
            [
                torch.sum(embeddings[predictions == i][:, c] > 0).item()
                for i in range(len(classes))
            ],
        )
        os.makedirs(f"{model}/{c}", exist_ok=True)
        tree.unfolding_tree(c.item(), f"{model}/{c}/tree.png")
        g_cnt = 0
        for i in range(len(data)):
            if embeddings[i, c] > 0:
                os.makedirs(f"{model}/{c}/{i}_y={data[i].y}", exist_ok=True)
                tree.color_unfolding_tree_in_graph(
                    c, data[i], f"{model}/{c}/{i}_y={data[i].y}"
                )
                g_cnt += 1
            if g_cnt >= 3:
                break
        cnt += 1
        if cnt >= 10:
            break

    print(sorted[:100])

    plt.hist(tree.weight.detach().numpy(), bins=100)
    plt.savefig(f"{model}/weight.png")
    print(
        f"non zero: {torch.sum(tree.weight > 0)}, zero: {torch.sum(tree.weight == 0)}"
    )
else:
    predictions = torch.load(
        os.path.join(
            GNN_DIR, info["dataset_name"], model, "3", "fold0", "predictions.pt"
        )
    ).flatten()
    labels = torch.tensor([g.y.item() for g in data])
    embeddings = torch.stack([tree.calc_subtree_weights(g) for g in data], dim=0)

    os.makedirs(model, exist_ok=True)
    cnt = 0
    for idx, c in enumerate(indices):
        if torch.sum(embeddings[:, c] > 0) < 10:
            continue
        counts = embeddings[:, c]
        print(
            f"{c.item()}: w={sorted[idx].item()}\
            r:{idx+1}\
            corr label:{torch.corrcoef(torch.stack([counts, labels], dim=0))[0, 1]}\
            corr pred: {torch.corrcoef(torch.stack([counts, predictions], dim=0))[0, 1]}"
        )
        os.makedirs(f"{model}/{c}", exist_ok=True)
        tree.unfolding_tree(c.item(), f"{model}/{c}/tree.png")
        plt.scatter(counts, labels)
        plt.savefig(f"{model}/{c}/scatter.png")
        plt.close()
        g_cnt = 0
        for i in range(len(data)):
            if embeddings[i, c] > 0:
                os.makedirs(f"{model}/{c}/{i}", exist_ok=True)
                tree.color_unfolding_tree_in_graph(c, data[i], f"{model}/{c}/{i}")
                g_cnt += 1
            if g_cnt >= 3:
                break
        cnt += 1
        if cnt >= 10:
            break

    print(
        f"0: w={tree.weight[0]}\
        r: ?\
        corr label:{torch.corrcoef(torch.stack([embeddings[:, 0], labels], dim=0))[0, 1]}\
        corr pred: {torch.corrcoef(torch.stack([embeddings[:, 0], predictions], dim=0))[0, 1]}"
    )
    tree.unfolding_tree(0, f"{model}/0.png")
    plt.scatter(embeddings[:, 0], labels)
    plt.savefig(f"{model}/0_scatter.png")
    plt.close()
    print(sorted[:100])

    plt.hist(tree.weight.detach().numpy(), bins=100)
    plt.savefig(f"{model}/weight.png")
    print(
        f"non zero: {torch.sum(tree.weight > 0)}, zero: {torch.sum(tree.weight == 0)}"
    )
