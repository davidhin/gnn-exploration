from pathlib import Path

import dgl
import gensim
import gnnproject as gp
import nltk
import numpy as np
import pandas as pd
import pygraphviz as pgv
import torch
from gnnproject.helpers.constants import EDGE_TYPES, TYPE_MAP, TYPE_MAP_OH
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
    cfgonly: bool = False,
):
    """Obtain DGL graph from code property graph output from Joern (old).

    EXAMPLE FOR DEBUGGING:
    import gnnproject as gp
    from gensim.models import Word2Vec
    sample = "devign_ffmpeg_qemu/21396_qemu_2caa9e9d2e0f356cc244bc41ce1d3e81663f6782_1"
    filepath = gp.processed_dir() / sample
    w2v = Word2Vec.load(str(gp.external_dir() / "w2v_models/devign"))
    etypemap = EDGE_TYPES_CD
    cfgonly = False
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

    # Filter edge types
    e = e[e.type.isin(etypemap)]

    def one_hot_encode_type(node_type):
        try:
            return TYPE_MAP_OH[TYPE_MAP[node_type] - 1].tolist()
        except:
            return -1

    if cfgonly:  # TODO Double check the ID mapping
        n.isCFGNode = n.isCFGNode.fillna(False)
        n = n[n.isCFGNode].copy()

    # Reset ID mappings
    node_id_dict = (
        n.reset_index(drop=True)
        .reset_index()[["key", "index"]]
        .set_index("key")
        .to_dict()["index"]
    )
    e = e[(e.start.isin(n.key)) & (e.end.isin(n.key))].copy()
    n.key = n.key.apply(lambda x: node_id_dict[x])
    e.start = e.start.apply(lambda x: node_id_dict[x])
    e.end = e.end.apply(lambda x: node_id_dict[x])

    # Embed features
    n.code = n.code.fillna("")
    n.code = n.code.apply(embed_code, w2v=w2v).to_list()
    n.type = n.type.apply(one_hot_encode_type)
    n = n[n.type != -1]

    # Format data appropriately
    src = e["start"].to_numpy()
    dst = e["end"].to_numpy()
    nnodes = len(n)
    nfeat = torch.tensor([list(i.type) + list(i.code) for i in n.itertuples()]).float()
    etype = torch.tensor([etypemap[i] for i in e.type]).int()

    try:
        label = int(label)
    except Exception as E:
        if verbose > 0:
            gp.debug(E)
        label = -1
    return (create_dgl_graph(src, dst, nnodes, nfeat, etype), label, filepath)
