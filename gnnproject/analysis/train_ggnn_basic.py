# %% SETUP
import datetime
import pickle as pkl
from collections import Counter
from glob import glob

import dgl
import gnnproject as gp
import gnnproject.helpers.dgl_helpers as dglh
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# CONSTANTS
DATASET = "devign_ffmpeg_qemu"  # Change to other datasets
BATCH_SIZE = 64

# %% Load own feature extracted graphs
dgl_proc_files = glob(str(gp.processed_dir() / f"{DATASET}_dgl/*"))
# dgl_proc_files = random.sample(dgl_proc_files, 1000)
train, val, test = dglh.train_val_test(dgl_proc_files)
print(len(train), len(val), len(test))
trainset = dglh.CustomGraphDataset(train)
valset = dglh.CustomGraphDataset(val)
testset = dglh.CustomGraphDataset(test)
gp.debug(Counter([int(i) for i in trainset.labels]))
gp.debug(Counter([int(i) for i in valset.labels]))
gp.debug(Counter([int(i) for i in testset.labels]))

# %% Load Reveal's Devign features
# trainset = dglh.RevealDevign("train.pkl")
# valset = dglh.RevealDevign("val.pkl", trainset.edge_type_dict)
# for i in valset:
#     if "h" in i[0].ndata:
#         i[0].ndata.pop("h")
# testset = dglh.RevealDevign("test.pkl", trainset.edge_type_dict)


# %% Get dataloader
dl_args = {"batch_size": BATCH_SIZE, "shuffle": True, "collate_fn": dglh.collate}
train_loader = DataLoader(trainset, **dl_args)
val_loader = DataLoader(valset, **dl_args)
test_loader = DataLoader(testset, **dl_args)

# %% Get DL model
model = dglh.BasicGGNN(169, 300, 2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
savedir = gp.get_dir(gp.processed_dir() / "dl_models")
savepath = savedir / "best_basic_ggnn.bin"
model = model.to("cuda")

# %% Start Tensorbaord
writer = SummaryWriter(
    savedir
    / "best_basic_ggnn"
    / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M_lr0.0001")
)


# %% Train DL model
model.train()
epoch_losses = []
best_f1 = 0
patience = 0
for epoch in range(100):
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

            loss = loss_func(output, label)
            tepoch.set_postfix(loss=loss.item())
            epoch_loss += loss.detach().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ...log the running loss
        epoch_loss /= iter + 1
        epoch_losses.append(epoch_loss)
        writer.add_scalar("Epoch Loss", epoch_loss, epoch * len(train_loader) + iter)

    scores = dglh.eval_model(model, val_loader)
    for s in scores.items():
        writer.add_scalar(s[0], s[1], epoch * len(train_loader) + iter)

    if scores["f1"] > best_f1:
        best_f1 = scores["f1"]
        with open(savepath, "wb") as f:
            torch.save(model.state_dict(), f)
        gp.debug(f"Best model saved. {scores} Patience: {patience}")
        patience = 0
    else:
        patience += 1
        gp.debug(f"No improvement. Patience: {patience}")


# %% Evaluate scores on splits
model.eval()
dglh.eval_model(model, train_loader)
dglh.eval_model(model, val_loader)
dglh.eval_model(model, test_loader)

# %% Get and save intermediate representations


def get_intermediate(model, data_loader):
    """Get second to last layer output of DL model."""
    rep = []
    labels = []

    def hook(module, input, output):
        input[0].ndata["features"] = output
        unbatched_g = dgl.unbatch(input[0])
        graph_reps = [
            dgl.mean_nodes(g, "features").detach().cpu().numpy() for g in unbatched_g
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


model.load_state_dict(torch.load(savepath))

dl_args = {"batch_size": 128, "shuffle": False, "collate_fn": dglh.collate}
train_loader = DataLoader(trainset, **dl_args)
val_loader = DataLoader(valset, **dl_args)
test_loader = DataLoader(testset, **dl_args)
train_graph_rep = get_intermediate(model, train_loader)
val_graph_rep = get_intermediate(model, val_loader)
test_graph_rep = get_intermediate(model, test_loader)


with open(gp.processed_dir() / "dl_models" / "basic_ggnn_hidden_train.pkl", "wb") as f:
    pkl.dump(train_graph_rep, f)
with open(gp.processed_dir() / "dl_models" / "basic_ggnn_hidden_val.pkl", "wb") as f:
    pkl.dump(val_graph_rep, f)
with open(gp.processed_dir() / "dl_models" / "basic_ggnn_hidden_test.pkl", "wb") as f:
    pkl.dump(test_graph_rep, f)
