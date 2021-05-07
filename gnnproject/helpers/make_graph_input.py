from pathlib import Path

import gnnproject as gp
import networkx
import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from networkx.readwrite import json_graph

EDGE_MAP = {
    "AST": 0,
    "CFG": 1,
    "CDG": 2,
    "DDG": 3,
    "PDG": 4,
}


def dot_to_json(file_in) -> dict:
    """Convert a dot file to json."""
    graph_netx = networkx.drawing.nx_pydot.read_dot(file_in)
    return json_graph.node_link_data(graph_netx)


def dot_to_node_edges(filepath: str) -> pd.DataFrame:
    """Get node and edge df given filepath to dot file."""
    jsongraph = dot_to_json(filepath)
    nodes = pd.DataFrame.from_records(jsongraph["nodes"])
    nodes["label"] = nodes["label"].apply(lambda x: x[2:-2].split(",", 2))
    nodes["label"] = nodes["label"].apply(lambda x: x + [""] if len(x) <= 2 else x)
    nodes[["f1", "f2", "f3"]] = pd.DataFrame(nodes.label.to_list())
    nodes = nodes.drop(columns=["label"])
    edges = pd.DataFrame.from_records(jsongraph["links"])
    edges["label"] = edges["label"].apply(lambda x: x.split(":")[0][1:])
    edges = edges.drop(columns=["key"])  # Not sure what Key means
    return nodes, edges


def get_gnn_input(filepath: str) -> dict:
    """Process a dot file so that it is suitable for input into a GNN.

    -----------------------------------------------------------------------
    NOT IN USE - SWITCHED TO OLD JOERN FOR CONSISTENCY WITH PREVIOUS PAPERS
    -----------------------------------------------------------------------

    Example:
    processed = glob(str(gp.processed_dir() / "**/*"))
    get_gnn_input(processed[1])
    >>> {'graph': array([[ 0,  0,  1],
    >>>     [ 0,  3,  1],
    >>>     [28,  3, 30],
    >>>     ...
    >>>     [28,  3, 30]]),
    >>> 'features': array([[-0.6487869 ,  0.8518523 , -0.56018966, ...,  0.98293424,
    >>>     0.50583929,  0.54708636],
    >>>     ...,
    >>>     [-0.63051212,  2.32098389, -7.08860016, ...,  1.29116583,
    >>>     1.06317401,  4.51795673]]),
    >>> 'target': '0',
    >>> 'file': '/home/.../6852_qemu_debaaa114a8877a939533ba846e64168fb287b7b_0.dot'}

    Args:
        filepath (str): Path to file.
    """
    # Extract nodes/edges into dataframe
    n, e = dot_to_node_edges(filepath)

    # Apply word2vec to obtain embeddings for code snippet
    # which act as the ingp.external_diitial node features
    n.f2 = n.f2.apply(lambda x: nltk.word_tokenize(x.strip()))
    w2vmodel = Word2Vec.load(str(gp.external_dir() / "w2v_models/devign"))

    def get_word_vector(w2v: Word2Vec, x: str) -> np.array:
        """Return word vector if in dictionary, else zero array."""
        if len(x) == 0:
            return np.zeros(100)
        try:
            return sum(np.array([w2v.wv[word] for word in x]))
        except:
            return np.zeros(100)

    n.f2 = n.f2.apply(lambda x: get_word_vector(w2vmodel, x))

    # Mapping tricks to ensure ID mappings for the edges
    # are consistent with the node ordering
    n = n.rename_axis("mapped_id").reset_index()
    node_mapping = n.set_index("id").to_dict()["mapped_id"]
    e.source = e.source.apply(lambda x: node_mapping[x])
    e.target = e.target.apply(lambda x: node_mapping[x])
    e.label = e.label.apply(lambda x: EDGE_MAP[x])

    # Return necessary information
    graph = e[["source", "label", "target"]].values
    features = np.array(n.f2.to_list())
    target = Path(filepath).stem.split("_")[-1]
    return {"graph": graph, "features": features, "target": target, "file": filepath}
