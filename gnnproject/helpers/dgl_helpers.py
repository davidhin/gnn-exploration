# %% SETUP
import pickle as pkl

import dgl
import gnnproject as gp
import matplotlib.pyplot as plt
import networkx as nx
import torch
from dgl.data import DGLDataset
from dgl.nn import GatedGraphConv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class CustomGraphDataset(DGLDataset):
    """Custom dataset loader."""

    def __init__(self, filepaths):
        """Initialise class."""
        self.filenames = []
        self.graphs = []
        self.labels = []
        for fp in tqdm(filepaths):
            with open(fp, "rb") as f:
                obj = pkl.load(f)
                self_loop_graph = dgl.add_self_loop(obj[0])
                self.graphs.append(self_loop_graph)
                self.labels.append(obj[1])
                self.filenames.append(obj[2])
        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        """Get item."""
        return self.graphs[i], self.labels[i]

    def __len__(self):
        """Get length."""
        return len(self.graphs)

    def get_filename(self, i):
        """Get filename of sample at index i."""
        return self.filenames[i]


class RevealDevign(DGLDataset):
    """Load devign dataset weights by ReVeal."""

    def __init__(self, split="train.pkl", edge_type_dict={}):
        """Initialise class."""
        self.edge_type_dict = edge_type_dict
        self.graphs = []
        self.labels = []
        with open(gp.external_dir() / "devign_ffmpeg_qemu" / split, "rb") as f:
            data = pkl.load(f)
        for g, l in tqdm(data):
            reset_edges = []
            g = dgl.add_self_loop(g)
            for e in g.edata["etype"].tolist():
                # print(g.edata["etype"])
                if e in self.edge_type_dict:
                    reset_edges += [self.edge_type_dict[e]]
                else:
                    reset_edges += [len(self.edge_type_dict)]
                    self.edge_type_dict[e] = len(self.edge_type_dict)
            g.edata["etype"] = torch.tensor(reset_edges)
            self.graphs.append(g)
            self.labels.append(l)
        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        """Get item."""
        return self.graphs[i], self.labels[i]

    def __len__(self):
        """Get length."""
        return len(self.graphs)

    def get_filename(self, i):
        """Get filename of sample at index i."""
        return self.filenames[i]


class BasicGGNN(nn.Module):
    """Basic GGNN for graph classification."""

    def __init__(self, in_dim, hidden_dim, n_classes, n_etypes=13):
        """Initialise."""
        super(BasicGGNN, self).__init__()
        self.ggnn = GatedGraphConv(
            in_feats=in_dim,
            out_feats=hidden_dim,
            n_steps=6,
            n_etypes=n_etypes,
        )
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        """Forward pass."""
        h = self.ggnn(g, g.ndata["_FEAT"], g.edata["_TYPE"])
        g.ndata["h"] = h
        hg = dgl.sum_nodes(g, "h")
        return self.classify(hg)


def collate(samples, device="cuda"):
    """Batch graphs."""
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_graph = batched_graph.to(device)
    labels = torch.tensor(labels)
    labels = labels.to(device)
    return batched_graph, labels


def plot_graph(input_graph):
    """Plot homogenous DGL graph."""
    graph, _ = input_graph
    label = 1
    _, ax = plt.subplots()
    nx.draw(graph.to_networkx(), with_labels=True, ax=ax)
    ax.set_title("Class: {:d}".format(label))
    plt.show()


def train_val_test(ilist: list, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10):
    """Split into train/val/test."""
    train, test = train_test_split(ilist, test_size=1 - train_ratio)
    val, test = train_test_split(test, test_size=test_ratio / (test_ratio + val_ratio))
    return train, val, test


def eval_model(model: nn.Module, data_loader: DataLoader):
    """Print evaluation metrics for model."""
    model.eval()
    with torch.no_grad():
        all_preds, all_targets = [], []
        with tqdm(data_loader) as tepoch:
            for bg, label in tepoch:
                output = model(bg)
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                all_preds += predictions.detach().cpu().tolist()
                all_targets += label.detach().cpu().tolist()
        eval_str = "Validation: "
        ret = {}
        for eval_met in zip(
            [accuracy_score, f1_score, precision_score, recall_score],
            ["acc", "f1", "prec", "rec"],
        ):
            score = round(eval_met[0](all_preds, all_targets), 2)
            eval_str += f"{eval_met[1]}: {score} | "
            ret[eval_met[1]] = score
        print("")
        gp.debug(eval_str)
    model.train()
    return ret
