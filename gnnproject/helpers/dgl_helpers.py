# %% SETUP
import pickle as pkl

import dgl
import gnnproject as gp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
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

    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_etypes=13,
        ndata_name="_FEAT",
        edata_name="_TYPE",
    ):
        """Initialise."""
        super(BasicGGNN, self).__init__()
        self.ggnn = GatedGraphConv(
            in_feats=in_dim,
            out_feats=hidden_dim,
            n_steps=6,
            n_etypes=n_etypes,
        )
        self.classify = nn.Linear(hidden_dim, 1)
        self.ndata_name = ndata_name
        self.edata_name = edata_name

    def forward(self, g):
        """Forward pass."""
        h = self.ggnn(g, g.ndata[self.ndata_name], g.edata[self.edata_name])
        h = F.relu(h)
        g.ndata["h"] = h
        hg = dgl.sum_nodes(g, "h")
        linearout = self.classify(hg)
        return torch.sigmoid(linearout).squeeze(dim=-1)


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


def train_val_test(
    ilist: list, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10, seed=0
):
    """Split into train/val/test."""
    train, test = train_test_split(ilist, test_size=1 - train_ratio, random_state=seed)
    if val_ratio == 0:
        return train, test
    val, test = train_test_split(
        test, test_size=test_ratio / (test_ratio + val_ratio), random_state=seed
    )
    return train, val, test


def eval_model(
    model: nn.Module, data_loader: DataLoader, loss_func, labels_to_float32: bool
):
    """Print evaluation metrics for model."""
    model.eval()
    with torch.no_grad():
        all_preds, all_targets = [], []
        loss = []
        with tqdm(data_loader) as tepoch:
            for bg, label in tepoch:
                output = model(bg)
                if labels_to_float32:
                    label = label.to(torch.float32)
                loss.append(loss_func(output, label).detach().cpu().item())
                predictions = (output > 0.5).float()
                all_preds += predictions.detach().cpu().tolist()
                all_targets += label.detach().cpu().tolist()
        eval_str = "Validation: "
        ret = {}
        ret["loss"] = np.mean(loss).item()
        eval_str += f"Loss: {round(np.mean(loss).item(), 4)} | "
        for eval_met in zip(
            [accuracy_score, f1_score, precision_score, recall_score],
            ["acc", "f1", "prec", "rec"],
        ):
            score = round(eval_met[0](all_preds, all_targets), 4)
            eval_str += f"{eval_met[1]}: {score} | "
            ret[eval_met[1]] = score
        print("")
        gp.debug(eval_str)
    model.train()
    return ret


def get_intermediate(model, data_loader):
    """Get second to last layer output of DL model."""
    rep = []
    labels = []

    def hook(module, input, output):
        input[0].ndata["features"] = output
        unbatched_g = dgl.unbatch(input[0])
        graph_reps = [
            dgl.sum_nodes(g, "features").detach().cpu().numpy() for g in unbatched_g
        ]
        rep.append(graph_reps)

    handle = model.ggnn.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        with tqdm(data_loader) as tepoch:
            for bg, label in tepoch:
                model(bg)
                labels += label.detach().cpu().tolist()

    handle.remove()
    rep = [i for j in rep for i in j]
    return list(zip(rep, labels))


def get_node_init_graph_features(dgl_proc_files: list, outprefix="no_ggnn", seed=0):
    """Get graph representations using node initialisations (ie no GGNN).

    EXAMPLE:
    DATASET = "devign_ffmpeg_qemu"  # Change to other datasets
    VARIATION = "cfgdfg"
    dgl_proc_files = glob(str(gp.processed_dir() / f"{DATASET}_dgl_{VARIATION}/*"))
    """

    def sum_node_inits(filepath):
        with open(filepath, "rb") as f:
            g = pkl.load(f)
            feat = np.sum(g[0].ndata["_FEAT"].numpy(), axis=0)
            label = g[1]
        return (feat, label, g[2])

    train, val, test = train_val_test(dgl_proc_files, seed=seed)
    train_noggnn = [sum_node_inits(i) for i in tqdm(train)]
    val_noggnn = [sum_node_inits(i) for i in tqdm(val)]
    test_noggnn = [sum_node_inits(i) for i in tqdm(test)]
    with open(gp.processed_dir() / "dl_models" / f"{outprefix}_train.pkl", "wb") as f:
        pkl.dump(train_noggnn, f)
    with open(gp.processed_dir() / "dl_models" / f"{outprefix}_val.pkl", "wb") as f:
        pkl.dump(val_noggnn, f)
    with open(gp.processed_dir() / "dl_models" / f"{outprefix}_test.pkl", "wb") as f:
        pkl.dump(test_noggnn, f)
    return train_noggnn, val_noggnn, test_noggnn
