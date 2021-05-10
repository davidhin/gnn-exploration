# %% LOAD IMPORTS
import sys
from glob import glob

import gnnproject as gp
import gnnproject.helpers.old_joern as gpj
import numpy as np

# %% SETUP
NUM_JOBS = 200
JOB_ARRAY_NUMBER = int(sys.argv[1]) - 1

# %% MAKE SPLITS
files = glob(str(gp.external_dir() / "devign_ffmpeg_qemu/functions/*"))
splits = np.array_split(files, NUM_JOBS)
savedir = gp.processed_dir() / "devign_ffmpeg_qemu"


# %% PROCESS SPLIT
def process_split(split: list):
    """Run joern on list of files sequentially."""
    for f in split:
        gpj.run_joern_old(
            f,
            "devign_ffmpeg_qemu",
            "old-joern-parse",
            save=True,
            verbose=1,
        )


process_split(splits[JOB_ARRAY_NUMBER])
