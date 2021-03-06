# %% SETUP
import argparse
import datetime
import json
import os
import pickle as pkl
import sys
from collections import Counter
from glob import glob
from pathlib import Path
from random import randrange

import gnnproject as gp
import gnnproject.helpers.dgl_helpers as dglh
import gnnproject.helpers.representation_learning as rlm
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main(args):
    """Train a basic GGNN."""
    # Load own feature extracted graphs
    dgl_proc_files = glob(
        str(gp.processed_dir() / f"{args.dataset}_dgl_{args.variation}/*")
    )
    ID = datetime.datetime.now().strftime(
        "%Y%m%d%H%M_{}_{}".format(
            gp.gitsha(), "_".join([f"{v}" for k, v in vars(args).items()])
        )
    )

    if args.noggnn:
        trainset, valset, testset = dglh.get_node_init_graph_features(
            dgl_proc_files, outprefix=f"basic_noggnn_{ID}"
        )
        rlearning_res_train, rlearning_res_test = rlm.representation_learning(
            gp.processed_dir() / "dl_models" / f"basic_noggnn_{ID}_train.pkl",
            gp.processed_dir() / "dl_models" / f"basic_noggnn_{ID}_test.pkl",
            no_ggnn=True,
        )
        final_savedir = gp.get_dir(gp.outputs_dir())
        with open(final_savedir / "basic_noggnn_results.csv", "a") as f:
            f.write(
                ",".join(
                    [
                        ID,
                        '"' + json.dumps(rlearning_res_train).replace('"', "'") + '"',
                        '"' + json.dumps(rlearning_res_test).replace('"', "'") + '"',
                    ]
                )
                + "\n"
            )
        sys.exit()

    # Save splits
    train, val, test = dglh.train_val_test(dgl_proc_files, seed=args.split_seed)
    records = []
    for s, n in [("train", train), ("val", val), ("test", test)]:
        records += [{"set": s, "file": Path(i).stem} for i in n]
    splitdir = gp.get_dir(gp.processed_dir() / "dl_models" / "splits")
    pd.DataFrame.from_records(records).to_csv(splitdir / f"{ID}.csv", index=0)

    # Load dataset
    cachedir = gp.get_dir(gp.interim_dir() / "cache")
    cachefile = cachedir / f"{args.dataset}_{args.variation}_{args.split_seed}.pkl"
    if os.path.exists(cachefile):
        try:
            gp.debug(f"Reading Cached result: {cachefile}")
            with open(cachefile, "rb") as f:
                trainset, valset, testset = pkl.load(f)
        except Exception as E:
            gp.debug(f"{E}: Probably because cache is currently being written to.")
            trainset = dglh.CustomGraphDataset(train)
            valset = dglh.CustomGraphDataset(val)
            testset = dglh.CustomGraphDataset(test)
    else:
        trainset = dglh.CustomGraphDataset(train)
        valset = dglh.CustomGraphDataset(val)
        testset = dglh.CustomGraphDataset(test)
        with open(cachefile, "wb") as f:
            pkl.dump([trainset, valset, testset], f)
    gp.debug(Counter([int(i) for i in trainset.labels]))
    gp.debug(Counter([int(i) for i in valset.labels]))
    gp.debug(Counter([int(i) for i in testset.labels]))

    # %Get dataloader
    dl_args = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "collate_fn": dglh.collate,
    }
    train_loader = DataLoader(trainset, **dl_args)
    val_loader = DataLoader(valset, **dl_args)
    test_loader = DataLoader(testset, **dl_args)

    # Get DL model
    if args.model == "ggnnsum":
        model = dglh.BasicGGNN(args.in_num, args.out_num)
    if args.model == "devign":
        model = dglh.DevignGGNN(args.in_num, args.out_num)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=0.001)
    savedir = gp.get_dir(gp.processed_dir() / "dl_models")
    savepath = savedir / f"best_basic_ggnn_{ID}.bin"
    model = model.to("cuda")

    # Start Tensorbaord
    writer = SummaryWriter(savedir / "best_basic_ggnn" / ID)

    # Train model
    dglh.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_func=loss_func,
        optimizer=optimizer,
        savepath=savepath,
        writer=writer,
        args=args,
    )
    torch.cuda.empty_cache()

    # Evaluate scores on splits
    model.load_state_dict(torch.load(savepath))
    ggnn_results_train = dglh.eval_model(model, train_loader, loss_func, True)
    ggnn_results_val = dglh.eval_model(model, val_loader, loss_func, True)
    ggnn_results_test = dglh.eval_model(model, test_loader, loss_func, True)

    # Get and save intermediate representations
    train_graph_rep = dglh.get_intermediate(model, train_loader)
    test_graph_rep = dglh.get_intermediate(model, test_loader)

    # %% Resample
    # val_graph_rep = dglh.get_intermediate(model, val_loader)
    # all_reps = train_graph_rep + val_graph_rep + test_graph_rep
    # train_graph_rep, test_graph_rep = dglh.train_val_test(
    #     all_reps, train_ratio=0.8, val_ratio=0, test_ratio=0.2, seed=args.split_seed
    # )

    with open(
        gp.processed_dir() / "dl_models" / f"basic_ggnn_{ID}_hidden_train.pkl", "wb"
    ) as f:
        pkl.dump(train_graph_rep, f)
    with open(
        gp.processed_dir() / "dl_models" / f"basic_ggnn_{ID}_hidden_test.pkl", "wb"
    ) as f:
        pkl.dump(test_graph_rep, f)

    # Get representation learning results
    rlearning_results_train, rlearning_results_test = rlm.representation_learning(
        gp.processed_dir() / "dl_models" / f"basic_ggnn_{ID}_hidden_train.pkl",
        gp.processed_dir() / "dl_models" / f"basic_ggnn_{ID}_hidden_test.pkl",
    )

    # Save results
    final_savedir = gp.get_dir(gp.outputs_dir())
    with open(final_savedir / "basic_ggnn_results.csv", "a") as f:
        f.write(
            ",".join(
                [
                    ID,
                    '"' + json.dumps(ggnn_results_train).replace('"', "'") + '"',
                    '"' + json.dumps(ggnn_results_val).replace('"', "'") + '"',
                    '"' + json.dumps(ggnn_results_test).replace('"', "'") + '"',
                    '"' + json.dumps(rlearning_results_train).replace('"', "'") + '"',
                    '"' + json.dumps(rlearning_results_test).replace('"', "'") + '"',
                ]
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="devign_ffmpeg_qemu",
        choices=["devign_ffmpeg_qemu"],
    )
    parser.add_argument("--variation", default="cfg", choices=["cfg", "cfgdfg", "cpg"])
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learn_rate", default=0.0001, type=float)
    parser.add_argument("--in_num", default=169, type=int)
    parser.add_argument("--out_num", default=200, type=int)
    parser.add_argument("--split_seed", default=randrange(5), type=int)
    parser.add_argument("--patience", default=30, type=int)
    parser.add_argument("--noggnn", action="store_true")
    parser.add_argument("--model", default="ggnnsum", choices=["devign", "ggnnsum"])
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    gp.debug(args)
    main(args)

# %%
