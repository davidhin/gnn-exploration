import argparse
import datetime
from glob import glob
from random import randrange

import gnnproject as gp
import gnnproject.helpers.dgl_helpers as dglh
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier


def main(args):
    """Train an MLP classifier on averaged node embeddings."""
    dgl_proc_files = glob(
        str(gp.processed_dir() / f"{args.dataset}_dgl_{args.variation}/*")
    )
    ID = datetime.datetime.now().strftime(
        "%Y%m%d%H%M_{}_{}".format(
            gp.gitsha(), "_".join([f"{v}" for k, v in vars(args).items()])
        )
    )

    trainset, valset, testset = dglh.get_node_init_graph_features(
        dgl_proc_files, outprefix=f"basic_noggnn_{ID}", seed=args.split_seed
    )

    trainset += valset
    X_train, y_train, _ = zip(*trainset)
    X_test, y_test, _ = zip(*testset)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    clf = MLPClassifier(random_state=1, max_iter=300, verbose=2).fit(X_train, y_train)
    gp.debug(accuracy_score(clf.predict(X_test), y_test))
    gp.debug(precision_score(clf.predict(X_test), y_test))
    gp.debug(recall_score(clf.predict(X_test), y_test))
    gp.debug(f1_score(clf.predict(X_test), y_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="devign_ffmpeg_qemu",
        choices=["devign_ffmpeg_qemu"],
    )
    parser.add_argument("--variation", default="cpg", choices=["cfg", "cfgdfg", "cpg"])
    parser.add_argument("--split_seed", default=randrange(5), type=int)
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    gp.debug(args)
    main(args)
