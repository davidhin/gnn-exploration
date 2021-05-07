# %%
import os
import shutil
import subprocess
from glob import glob
from pathlib import Path

import gnnproject as gp

# %%


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


def run_joern_old(file_in: str, dataset: str):
    """Run joern (old version) on a given file. :TODO: Modularise so that it gets
    the output as node / edge df, instead of saving it.

    Args:
        file_in (str): [description]
        dataset (str): [description]
    """
    filename = Path(file_in).stem
    save_out = gp.get_dir(gp.processed_dir() / dataset / filename)

    if os.path.exists(save_out):
        gp.debug(f"Already completed {filename}")
        return

    tmppath = gp.get_dir(gp.interim_dir() / str(dataset + "_oldjoern") / filename)
    shutil.copy(file_in, tmppath)

    subprocess_cmd(
        f"cd {tmppath}; mkdir tmp; mv *.c tmp; singularity exec {gp.project_dir() / 'joernold.simg'} old-joern-parse tmp"
    )

    for f in glob(f"{tmppath}/parsed/tmp/{Path(file_in).name}/*"):
        shutil.move(f, save_out)

    shutil.rmtree(tmppath)


files = glob(str(gp.external_dir() / "devign_ffmpeg_qemu/functions/*"))

run_joern_old(files[0], dataset)

# %%


def run_joern(
    filepath: str,
    dataset_name: str,
    save: bool = False,
    joern_parse: str = "joern-parse",
    joern_export: str = "joern-export",
):
    """Get output of Joern-parse given filepath.

    -----------------------------------------------------------------------
    NOT IN USE - SWITCHED TO OLD JOERN FOR CONSISTENCY WITH PREVIOUS PAPERS
    -----------------------------------------------------------------------

    Example:
    joern_on_file(
        filepath=gp.external_dir() / "reveal_chrome_debian/functions/0_0.c",
        dataset_name="reveal_chrome_debian",
    )
    >>> digraph curl_mvsprintf {
    >>> "1000100" [label = "(METHOD,curl_mvsprintf)" ]
    >>> "1000101" [label = "(PARAM,char * buffer)" ]
    >>> "1000102" [label = "(PARAM,const char * format)" ]
    >>> "1000103" [label = "(PARAM,va_list ap_save)" ]
    >>> "1000104" [label = "(BLOCK,,)" ]

    Args:
        filepath (str): Full path to file
        dataset_name (str): Name of dataset (only used for directory organisation)
        joern_parse (str): Path to joern-parse executable
        joern_export (str): Path to joern_export executable
        save (bool, optional): Save to file or return raw output. Defaults to False.
    """
    filename = str(filepath).split("/")[-1].split(".")[0]
    savepath_interim = gp.get_dir(gp.interim_dir() / dataset_name)
    savepath_processed = gp.get_dir(gp.processed_dir() / dataset_name)
    joern_parse = joern_parse.split()
    joern_export = joern_export.split()

    try:
        subprocess.run(
            joern_parse
            + [
                "--out={}.bin".format(savepath_interim / filename),
                filepath,
            ],
        )

        subprocess.run(
            joern_export
            + [
                "--repr=cpg14",
                savepath_interim / "{}.bin".format(filename),
                "--out={}".format(savepath_interim / filename),
            ],
        )

        # Read first CPG : I think this is the main CPG, and the others are for func calls
        with open(savepath_interim / filename / "0-cpg.dot") as f:
            joern_output = f.read()
    except Exception as E:
        return "FAILED: {} - {}".format(filename, E)

    # Delete interim files
    os.remove(savepath_interim / "{}.bin".format(filename))
    shutil.rmtree(savepath_interim / filename)

    if save:
        final_path = savepath_processed / "{}.dot".format(filename)
        with open(final_path, "w") as f:
            joern_output = f.write(joern_output)
        return "Saved to {}".format(final_path)
    else:
        return joern_output
