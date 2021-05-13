from pathlib import Path

import dgl
import gensim
import gnnproject as gp
import nltk
import numpy as np
import pandas as pd
import pygraphviz as pgv
import torch
from gnnproject.helpers.constants import EDGE_TYPES
from IPython.display import Image, display


def format_node_edges(n, e):
    """Format node and edges into appropriate input form."""
    nodes = n.copy()
    edges = e.copy()
    nodes.key -= 1
    edges.start -= 1
    edges.end -= 1
    node_key_type_map = nodes[["key", "type"]].set_index("key").to_dict()["type"]
    node_key_code_map = nodes[["key", "code"]].set_index("key").to_dict()["code"]
    edges["src"] = edges.start.apply(lambda x: node_key_type_map[x])
    edges["dest"] = edges.end.apply(lambda x: node_key_type_map[x])
    edges["src_feat"] = edges.start.apply(lambda x: node_key_code_map[x])
    edges["dest_feat"] = edges.end.apply(lambda x: node_key_code_map[x])
    return nodes, edges


def plot_heterograph(edges, outdir="graph.png"):
    """Plot a heterograph given edges df parsed by format_node_edges()."""
    ag = pgv.AGraph(strict=False, directed=True)
    for e in edges.itertuples():
        ag.add_edge(
            f"ID{e.start}\n{e.src}\n{e.src_feat}",
            f"ID{e.end}\n{e.dest}\n{e.dest_feat}",
            label=f"{e.type}",
        )
    ag.layout("dot")
    ag.draw(outdir)
    display(Image(filename=outdir))
    gp.debug(f"Saved to {outdir}")


def plot_heterograph_from_filepath(filepath: str, outdir="graph.png"):
    """Plot heterograph given filepath input."""
    nodes = pd.read_csv(filepath / "nodes.csv", sep="\t")
    edges = pd.read_csv(filepath / "edges.csv", sep="\t")
    _, e = format_node_edges(nodes, edges)
    plot_heterograph(e, outdir)


def create_dgl_graph(
    src: np.array, dst: np.array, nnodes: int, nfeat: torch.tensor, etype: torch.tensor
) -> dgl.graph:
    """Create a graph with features and edge types embedded."""
    g = dgl.graph((src, dst), num_nodes=nnodes)
    g.ndata["_FEAT"] = nfeat
    g.edata["_TYPE"] = etype
    return g


def embed_code(code: str, w2v: gensim.models.word2vec) -> np.array:
    """Embed code using given word2vec model by averaging code embeddings."""
    code = nltk.word_tokenize(code.strip())
    if len(code) == 0:
        return np.zeros(100)
    try:
        return np.array(
            sum(np.array([w2v.wv[word] for word in code])) / len(code), dtype="float32"
        )
    except:
        return np.zeros(100)


def cpg_to_dgl_from_filepath(
    filepath: Path,
    w2v: gensim.models.word2vec,
    etypemap: map = EDGE_TYPES,
    verbose: int = 0,
):
    """Obtain DGL graph from code property graph output from Joern (old).

    EXAMPLE FOR DEBUGGING:
    import gnnproject as gp
    from gensim.models import Word2Vec
    sample = "devign_ffmpeg_qemu/21396_qemu_2caa9e9d2e0f356cc244bc41ce1d3e81663f6782_1"
    filepath = gp.processed_dir() / sample
    w2v = Word2Vec.load(str(gp.external_dir() / "w2v_models/devign"))
    etypemap = EDGE_TYPES
    """
    try:
        nodes = pd.read_csv(filepath / "nodes.csv", sep="\t")
        edges = pd.read_csv(filepath / "edges.csv", sep="\t")
    except Exception as E:
        if verbose > 0:
            gp.debug(E)
        return None
    if len(nodes) == 0 or len(edges) == 0:
        if verbose > 0:
            gp.debug("Empty node / edge CSV")
        return None
    label = str(filepath).split("_")[-1]
    n, e = format_node_edges(nodes, edges)
    e = e[e.type != "IS_FILE_OF"]
    src = e["start"].to_numpy()
    dst = e["end"].to_numpy()
    nnodes = len(n)
    n.code = n.code.fillna("")
    nfeat = torch.tensor(n.code.apply(embed_code, w2v=w2v).to_list()).float()
    etype = torch.tensor([etypemap[i] for i in e.type])

    try:
        label = int(label)
    except Exception as E:
        if verbose > 0:
            gp.debug(E)
        label = -1
    return (create_dgl_graph(src, dst, nnodes, nfeat, etype), label, filepath)
