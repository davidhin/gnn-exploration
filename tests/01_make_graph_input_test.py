# %%
from glob import glob

import gnnproject as gp
import gnnproject.helpers.make_graph_input as gpgi
import numpy as np


def test_dot_to_json():
    """Test dot_to_json returns a valid json output."""
    processed = glob(str(gp.processed_dir() / "**/*"))
    out = gpgi.dot_to_json(processed[0])
    assert type(out) is dict
    assert "graph" in out.keys()
    assert "links" in out.keys()


def test_dot_to_node_edges():
    """Test dot to node edges returns valid dataframes."""
    processed = glob(str(gp.processed_dir() / "**/*"))
    out = gpgi.dot_to_node_edges(processed[0])
    assert len(out) == 2, "{} : Dot to node edges should return two dataframes".format(
        len(out)
    )
    assert all(
        out[0].columns == ["id", "f1", "f2", "f3"]
    ), "{} should be id, f1, f2, f3".format(out[1].columns)

    assert all(
        out[1].columns == ["label", "source", "target"]
    ), "{} should be label, source, target".format(out[1].columns)
    assert len(out[0]) > 0, "Node dataframe length is zero, which is weird."
    assert len(out[1]) > 0, "Edge dataframe length is zero, which is weird."


def test_get_gnn_input():
    """Test output of get_gnn_input is valid."""
    processed = glob(str(gp.processed_dir() / "**/*"))
    out = gpgi.get_gnn_input(processed[1])
    assert list(out.keys()) == [
        "graph",
        "features",
        "target",
        "file",
    ], "{} keys are invalid.".format(out.keys())
    assert out["target"] == "1" or out["target"] == "0"
    assert len(out["graph"][0]) == 3, "graph items should be [source, edge, target]"
    assert (
        len(out["features"][0]) >= 100
    ), "word2vec should be >= 100, otherwise something went wrong"
    assert ".dot" in out["file"]
    assert all(
        [type(i) == np.ndarray for i in out["graph"]]
    ), "All graph items should be np.ndarray"
    assert all(
        [type(i) == np.ndarray for i in out["features"]]
    ), "All features should be np.ndarray"
