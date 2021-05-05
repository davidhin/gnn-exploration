# %% LOAD IMPORTS
import pickle as pkl
import sys
from glob import glob
from pathlib import Path

import gnnproject as gp
import gnnproject.helpers.make_graph_input as gpgi
import numpy as np

# %% SETUP
NUM_JOBS = 200
JOB_ARRAY_NUMBER = int(sys.argv[1]) - 1

# %% MAKE SPLITS
files = glob(str(gp.processed_dir() / "devign_ffmpeg_qemu/*"))
splits = np.array_split(files, NUM_JOBS)
savedir = gp.get_dir(gp.processed_dir() / "devign_ffmpeg_qemu_graph_input")


# %% PROCESS SPLIT
def process_split(split: list):
    """Process list of files sequentially."""
    for f in split:
        done = [Path(i).stem for i in glob(str(savedir / "*"))]
        if Path(f).stem in done:
            gp.debug("Already finished, skipping {}".format(Path(f).stem))
            continue
        out = gpgi.get_gnn_input(f)
        while True:
            try:
                pkl.dump(out, open(savedir / (Path(f).stem + ".bin"), "wb"))
                pkl.load(open(savedir / (Path(f).stem + ".bin"), "rb"))
            except Exception as E:
                gp.debug("{}, retrying...".format(E))
            else:
                break
        gp.debug("Finished {}".format(Path(f).stem))


process_split(splits[JOB_ARRAY_NUMBER])
