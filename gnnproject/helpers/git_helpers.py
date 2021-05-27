import os

import gnnproject as gp


def clone_repo(repo: str):
    """Clone repo given repo name in form a/b."""
    savepath = gp.external_dir() / "repos" / repo
    if os.path.exists(savepath):
        return savepath
    gp.get_dir(savepath)
    gp.subprocess_cmd(f"git clone https://github.com/{repo}.git {savepath}")
    return savepath


def get_commit_message(repo: str, sha: str):
    """Get commit message given reponame and commit sha."""
    repodir = clone_repo(repo) / ".git"
    ret = gp.subprocess_cmd(
        f"git --git-dir {repodir}  rev-list --format=%B --max-count=1 {sha}"
    )
    return ret[0].decode()


def get_lines_changed(repo: str, sha: str):
    """Get lines changed."""
    repodir = clone_repo(repo) / ".git"
    ret = gp.subprocess_cmd(f"git --git-dir {repodir}  diff --shortstat {sha}^ {sha}")
    ret = ret[0].decode().split(",")[1:]
    ret = [int(i.strip().split()[0]) for i in ret]
    return ret
