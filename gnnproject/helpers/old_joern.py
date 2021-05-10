import os
import shutil
import subprocess
from glob import glob
from pathlib import Path

import gnnproject as gp
import pandas as pd


def subprocess_cmd(command: str, verbose: int = 0):
    """Run command line process.

    Example:
    subprocess_cmd('echo a; echo b', verbose=1)
    >>> a
    >>> b
    """
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output = process.communicate()
    if verbose > 0:
        gp.debug(output[0].decode())
        gp.debug(output[1].decode())


def run_joern_old(
    file_in: str, dataset: str, joern_old_path: str, save: bool = True
) -> pd.DataFrame:
    """Run joern (old version) on a given file.

    Args:
        file_in (str): Path to .c file
        dataset (str): String for save-out path (not really important)
        joern_old_path (str): Path to joern_old executable.
        save (bool): Whether to save the data to storage/processed or return the df.
    """
    filename = Path(file_in).stem
    save_out = gp.processed_dir() / dataset / filename
    if save:
        gp.get_dir(save_out)

    if os.path.exists(save_out / "nodes.csv"):
        gp.debug(f"Already completed {filename}")
        if not save:
            return [
                pd.read_csv(save_out / "nodes.csv", sep="\t"),
                pd.read_csv(save_out / "edges.csv", sep="\t"),
            ]
        return

    tmppath = gp.get_dir(gp.interim_dir() / str(dataset + "_oldjoern") / filename)
    shutil.copy(file_in, tmppath)

    subprocess_cmd(f"cd {tmppath}; mkdir tmp; mv *.c tmp; {joern_old_path} tmp")

    ret = []
    for f in glob(f"{tmppath}/parsed/tmp/{Path(file_in).name}/*"):
        ret.append(pd.read_csv(f, sep="\t"))
        if save:
            shutil.move(f, save_out)

    shutil.rmtree(tmppath)
    if not save:
        return ret
    return
