# %%
import multiprocessing as mp
from glob import glob
from pathlib import Path

import gnnproject as gp
import gnnproject.helpers.git_helpers as ggh
import pandas as pd
from tqdm import tqdm

pd.set_option("max_colwidth", 500)

devign = [
    Path(i).stem.split("_")[1:-1]
    for i in glob(str(gp.external_dir() / "devign_ffmpeg_qemu/functions/*"))
]


def mp_commit_messages(item):
    """Parallelise get commit messages."""
    repo = f"{item[0]}/{item[0]}".lower()
    commit = item[1]
    return ggh.get_commit_message(repo, commit)


def mp_lines_changed(item):
    """Parallelise get commit messages."""
    repo = f"{item[0]}/{item[0]}".lower()
    commit = item[1]
    return ggh.get_lines_changed(repo, commit)


c_messages = []
with mp.Pool(mp.cpu_count()) as p:
    for i in tqdm(p.imap(mp_commit_messages, devign), total=len(devign)):
        c_messages.append(i)

lines_changed = []
with mp.Pool(mp.cpu_count()) as p:
    for i in tqdm(p.imap(mp_lines_changed, devign), total=len(devign)):
        lines_changed.append(i)

# %% Create dataframe
devign_df = pd.DataFrame(devign, columns=["repo", "commit"])
devign_df["cm"] = c_messages
devign_df["lc"] = lines_changed
devign_df["cm"] = devign_df["cm"].apply(lambda x: x[48:])
devign_df["lc"] = devign_df["lc"].apply(sum)

# %% Search
devign_df.cm = devign_df.cm.str.lower()
devign_df[devign_df.cm.str.contains("missing")].sort_values("lc").head(20)

# %%
