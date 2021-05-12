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
    while True:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )
        output = process.communicate()
        if "old-joern-parse: not found" in output[1].decode():
            command = command.replace(
                "old-joern-parse",
                f"singularity exec {gp.project_dir()}/main.simg old-joern-parse",
            )
            if verbose > 2:
                gp.debug("ERROR: old-joern-parse not in path. Running from image...")
                gp.debug(command)
        else:
            break
    if verbose > 1:
        gp.debug(output[0].decode())
        gp.debug(output[1].decode())


def run_joern_old(
    file_in: str, dataset: str, joern_old_path: str, save: bool = True, verbose=0
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
        if verbose > 0:
            gp.debug(f"Already completed {filename}")
        if not save:
            return [
                pd.read_csv(save_out / "nodes.csv", sep="\t"),
                pd.read_csv(save_out / "edges.csv", sep="\t"),
            ]
        return

    tmppath = gp.get_dir(gp.interim_dir() / str(dataset + "_oldjoern") / filename)
    shutil.copy(file_in, tmppath)

    subprocess_cmd(
        f"cd {tmppath}; mkdir tmp; mv *.c tmp; {joern_old_path} tmp", verbose=verbose
    )

    ret = []
    for f in glob(f"{tmppath}/parsed/tmp/{Path(file_in).name}/*"):
        ret.append(pd.read_csv(f, sep="\t"))
        if save:
            shutil.move(f, save_out)
    if verbose > 0:
        gp.debug(f"Saved to {save_out}")

    shutil.rmtree(tmppath)
    if not save:
        return ret
    return
