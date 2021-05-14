import dgl
import gnnproject as gp
import gnnproject.helpers.make_graph_input_oj as ggi
import gnnproject.helpers.old_joern as gpj
from gensim.models import Word2Vec
from gnnproject.helpers.constants import EDGE_TYPES_CD


def test_joern_graph_output():
    """Test devign test.c file and output a graph in outputs folder."""
    input_file = gp.project_dir() / "tests/devign_basic_c_func.c"
    n, e = gpj.run_joern_old(input_file, "", "old-joern-parse", save=False)
    n, e = ggi.format_node_edges(n, e)
    ggi.plot_heterograph(e, outdir=str(gp.outputs_dir() / "devign_basic_c_func.png"))
    assert len(n) > 20
    assert len(e) > 20


def test_cpg_to_dgl_from_filepath():
    """Test CPG to DGL function from filepath."""
    input_file = gp.project_dir() / "tests/devign_basic_c_func.c"
    gpj.run_joern_old(input_file, "", "old-joern-parse", save=True)
    processed_path = gp.processed_dir() / "devign_basic_c_func"
    w2vmodel = Word2Vec.load(str(gp.external_dir() / "w2v_models/devign"))
    g = ggi.cpg_to_dgl_from_filepath(processed_path, w2vmodel)
    assert type(g[0]) == dgl.DGLGraph
    assert type(g[1]) == int
    assert "_FEAT" in g[0].ndata
    assert "_TYPE" in g[0].edata
    assert len(g[0].ndata["_FEAT"]) > 10
    assert len(g[0].edata["_TYPE"]) > 10


def test_joern_graph_output_simple():
    """Test devign test.c file and output a graph in outputs folder."""
    input_file = gp.project_dir() / "tests/devign_basic_c_func.c"
    n, e = gpj.run_joern_old(input_file, "", "old-joern-parse", save=False)
    n, e = ggi.format_node_edges(n, e)

    n.isCFGNode = n.isCFGNode.fillna(False)
    n = n[n.isCFGNode].copy()
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

    e = e[e.type.isin(EDGE_TYPES_CD)]
    ggi.plot_heterograph(
        e, outdir=str(gp.outputs_dir() / "devign_basic_c_func_cfgdfg_simple.png")
    )
