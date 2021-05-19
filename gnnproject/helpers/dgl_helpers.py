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
from torch.nn.utils.rnn import pad_sequence
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

    def get_graph_embeddings(self, g):
        """Get graph embedding for a batched graph."""
        h = self.ggnn(g, g.ndata[self.ndata_name], g.edata[self.edata_name])
        g.ndata["h"] = h
        embeddings = []
        for gi in dgl.unbatch(g):
            embeddings.append(dgl.sum_nodes(gi, "h").detach().cpu().numpy())
        return embeddings


def unbatch_graph_to_tensor(g, ndata_name: str):
    """Given a batched graph, unbatch and return a tensor.

    Output tensor is of shape: (BATCH_SIZE, FEATURE_SIZE, NUM_NODES)
    """
    graphs = dgl.unbatch(g)
    batch_input = pad_sequence([g.ndata[ndata_name] for g in graphs])
    batch_input = batch_input.transpose(0, 1).transpose(1, 2)
    return batch_input


class DevignGGNN(nn.Module):
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
        super(DevignGGNN, self).__init__()
        self.ggnn = GatedGraphConv(
            in_feats=in_dim,
            out_feats=hidden_dim,
            n_steps=6,
            n_etypes=n_etypes,
        )
        self.ndata_name = ndata_name
        self.edata_name = edata_name

        self.conv_l1 = torch.nn.Conv1d(hidden_dim, hidden_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = in_dim + hidden_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, g):
        """Forward pass."""
        x = g.ndata[self.ndata_name]
        h = self.ggnn(g, x, g.edata[self.edata_name])
        g.ndata["h"] = h
        g.ndata["c"] = torch.cat((h, x), dim=-1)

        batched_h = unbatch_graph_to_tensor(g, "h")
        Y_1 = self.maxpool1(F.relu(self.conv_l1(batched_h)))
        Y_2 = self.maxpool2(F.relu(self.conv_l2(Y_1)))
        Y_2 = Y_2.transpose(1, 2)

        batched_c = unbatch_graph_to_tensor(g, "c")
        Z_1 = self.maxpool1_for_concat(F.relu(self.conv_l1_for_concat(batched_c)))
        Z_2 = self.maxpool2_for_concat(F.relu(self.conv_l2_for_concat(Z_1)))
        Z_2 = Z_2.transpose(1, 2)

        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = torch.sigmoid(avg).squeeze(dim=-1)
        return result


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
    model: nn.Module,
    data_loader: DataLoader,
    loss_func,
    labels_to_float32: bool,
    verbose=0,
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
            score = round(eval_met[0](all_targets, all_preds), 4)
            eval_str += f"{eval_met[1]}: {score} | "
            ret[eval_met[1]] = score
        print("")
        gp.debug(eval_str)
        if verbose > 1:
            gp.debug(all_preds[:50])
            gp.debug(all_targets[:50])
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


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_func,
    optimizer,
    savepath: str,
    writer,
    args,
):
    """Train DL model."""
    model.train()
    epoch_losses = []
    best_score = 0
    patience = 0
    for epoch in range(500):
        epoch_loss = 0
        with tqdm(train_loader) as tepoch:
            for iter, (bg, label) in enumerate(tepoch):
                if len(epoch_losses) > 0:
                    tepoch.set_description(
                        f"Epoch {epoch} (loss: {round(epoch_losses[-1], 4)})"
                    )
                else:
                    tepoch.set_description(f"Epoch {epoch}")

                output = model(bg)

                loss = loss_func(output, label.to(torch.float32))
                tepoch.set_postfix(loss=loss.item())
                epoch_loss += loss.detach().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ...log the running loss
            epoch_loss /= iter + 1
            epoch_losses.append(epoch_loss)
            writer.add_scalar(
                "Epoch Loss", epoch_loss, epoch * len(train_loader) + iter
            )

        scores = eval_model(model, val_loader, loss_func, True)
        writer.add_scalar("Val Loss", scores["loss"], epoch * len(train_loader) + iter)
        for s in scores.items():
            writer.add_scalar(s[0], s[1], epoch * len(train_loader) + iter)

        if scores["f1"] > best_score:
            best_score = scores["f1"]
            with open(savepath, "wb") as f:
                torch.save(model.state_dict(), f)
            gp.debug(f"Best model saved. {scores} Patience: {patience}")
            patience = 0
        else:
            patience += 1
            gp.debug(f"No improvement. Patience: {patience}")

        if patience > args.patience:
            gp.debug("Training Complete.")
            break
