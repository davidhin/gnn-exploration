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
VARIATION = "cfgdfg"
BATCH_SIZE = 64
LEARN_RATE = 0.0001
IN_NUM = 169
OUT_NUM = 200
SPLIT_SEED = 0

# %% Load own feature extracted graphs
dgl_proc_files = glob(str(gp.processed_dir() / f"{DATASET}_dgl_{VARIATION}/*"))
train, val, test = dglh.train_val_test(dgl_proc_files, seed=SPLIT_SEED)
print(len(train), len(val), len(test))
trainset = dglh.CustomGraphDataset(train)
valset = dglh.CustomGraphDataset(val)
testset = dglh.CustomGraphDataset(test)
gp.debug(Counter([int(i) for i in trainset.labels]))
gp.debug(Counter([int(i) for i in valset.labels]))
gp.debug(Counter([int(i) for i in testset.labels]))

# %% Get dataloader
dl_args = {"batch_size": BATCH_SIZE, "shuffle": True, "collate_fn": dglh.collate}
train_loader = DataLoader(trainset, **dl_args)
val_loader = DataLoader(valset, **dl_args)
test_loader = DataLoader(testset, **dl_args)

# %% Get DL model
model = dglh.BasicGGNN(IN_NUM, OUT_NUM, 2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=0.001)
savedir = gp.get_dir(gp.processed_dir() / "dl_models")
ID = datetime.datetime.now().strftime(
    "%Y%m%d%H%M_{}_{}_{}_{}".format(DATASET, VARIATION, LEARN_RATE, OUT_NUM)
)
savepath = savedir / f"best_basic_ggnn_{ID}.bin"
model = model.to("cuda")

# %% Start Tensorbaord
writer = SummaryWriter(savedir / "best_basic_ggnn" / ID)


# %% Train DL model
model.train()
epoch_losses = []
best_f1 = 0
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

    if patience > 50:
        gp.debug("Training Complete.")
        break

# %% Evaluate scores on splits
model.load_state_dict(torch.load(savepath))
gp.debug(dglh.eval_model(model, train_loader))
gp.debug(dglh.eval_model(model, val_loader))
gp.debug(dglh.eval_model(model, test_loader))

# %% Get and save intermediate representations
dl_args = {"batch_size": 128, "shuffle": False, "collate_fn": dglh.collate}
train_loader = DataLoader(trainset, **dl_args)
val_loader = DataLoader(valset, **dl_args)
test_loader = DataLoader(testset, **dl_args)
train_graph_rep = dglh.get_intermediate(model, train_loader)
val_graph_rep = dglh.get_intermediate(model, val_loader)
test_graph_rep = dglh.get_intermediate(model, test_loader)


with open(
    gp.processed_dir() / "dl_models" / f"basic_ggnn_{ID}_hidden_train.pkl", "wb"
) as f:
    pkl.dump(train_graph_rep, f)
with open(
    gp.processed_dir() / "dl_models" / f"basic_ggnn_{ID}_hidden_val.pkl", "wb"
) as f:
    pkl.dump(val_graph_rep, f)
with open(
    gp.processed_dir() / "dl_models" / f"basic_ggnn_{ID}_hidden_test.pkl", "wb"
) as f:
    pkl.dump(test_graph_rep, f)
