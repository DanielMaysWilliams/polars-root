import pytest
import polars as pl
import numpy as np
import uproot

from polars_root import read_root


@pytest.fixture
def simple_root_file(tmp_path):
    # Create a ROOT file with a simple TTree using uproot
    file_path = tmp_path / "test.root"
    data = {
        "x": np.array([1, 2, 3], dtype=np.int32),
        "y": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        "z": np.array(["a", "b", "c"], dtype="U1"),
    }
    with uproot.recreate(file_path) as f:
        f["tree"] = data
    return str(file_path), "tree", data


def test_scan_root_reads_simple_tree(simple_root_file):
    file_path, tree_name, data = simple_root_file
    df = read_root(file_path, tree_name)
    assert df.shape == (3, 3)
    assert df["x"].to_list() == data["x"].tolist()
    assert df["y"].to_list() == data["y"].tolist()
    assert df["z"].to_list() == list(data["z"])


def test_scan_root_schema(simple_root_file):
    file_path, tree_name, _ = simple_root_file
    df = read_root(file_path, tree_name)
    schema = df.collect_schema()
    assert set(schema.keys()) == {"x", "y", "z"}
    assert schema["x"] == pl.Int32
    assert schema["y"] == pl.Float64
    assert schema["z"] == pl.String


def test_scan_root_with_predicate(simple_root_file):
    file_path, tree_name, _ = simple_root_file
    df = read_root(file_path, tree_name)
    filtered = df.filter(pl.col("x") > 1)
    assert filtered.shape == (2, 3)
    assert filtered["x"].to_list() == [2, 3]


def test_scan_root_invalid_tree(tmp_path):
    # Create a ROOT file with no tree
    file_path = tmp_path / "empty.root"
    with uproot.recreate(file_path):
        pass
    with pytest.raises(Exception):
        read_root(str(file_path), "not_a_tree")


def test_scan_root_colon_tree_name(simple_root_file):
    file_path, tree_name, data = simple_root_file
    # Should work if tree_name is None and file has only one tree
    df = read_root(file_path + ":" + tree_name)
    assert df.shape == (3, 3)
    assert df["x"].to_list() == data["x"].tolist()


def test_scan_root_invalid_tree_name(simple_root_file):
    file_path, _, _ = simple_root_file
    with pytest.raises(Exception):
        read_root(file_path)
