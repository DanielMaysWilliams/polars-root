import polars as pl
import pytest
import uproot

from polars_root import scan_root


def test_scan_root_reads_one_column_tree(one_column_root_file):
    file_path, tree_name, data = one_column_root_file
    lf = scan_root(file_path, tree_name)
    df = lf.collect()
    assert df.shape == (3, 1)
    assert df["x"].to_list() == data["x"].tolist()


def test_scan_root_reads_simple_tree(simple_root_file):
    file_path, tree_name, data = simple_root_file
    lf = scan_root(file_path, tree_name)
    df = lf.collect()
    assert df.shape == (3, 3)
    assert df["x"].to_list() == data["x"].tolist()
    assert df["y"].to_list() == data["y"].tolist()
    assert df["z"].to_list() == list(data["z"])


def test_scan_root_schema(simple_root_file):
    file_path, tree_name, data = simple_root_file
    lf = scan_root(file_path, tree_name)
    schema = lf.collect_schema()
    assert set(schema.keys()) == {"x", "y", "z"}
    assert schema["x"] == pl.Int32
    assert schema["y"] == pl.Float64
    assert schema["z"] == pl.String


def test_scan_root_with_predicate(simple_root_file):
    file_path, tree_name, data = simple_root_file
    lf = scan_root(file_path, tree_name)
    filtered = lf.filter(pl.col("x") > 1).collect()
    assert filtered.shape == (2, 3)
    assert filtered["x"].to_list() == [2, 3]


def test_scan_root_invalid_tree(tmp_path):
    # Create a ROOT file with no tree
    file_path = tmp_path / "empty.root"
    with uproot.recreate(file_path):
        pass
    with pytest.raises(KeyError):
        scan_root(str(file_path), "not_a_tree")


def test_scan_root_colon_tree_name(simple_root_file):
    file_path, tree_name, data = simple_root_file
    # Should work if tree_name is None and file has only one tree
    lf = scan_root(file_path + ":" + tree_name)
    df = lf.collect()
    assert df.shape == (3, 3)
    assert df["x"].to_list() == data["x"].tolist()


def test_scan_root_invalid_tree_name(simple_root_file):
    file_path, _, _ = simple_root_file
    with pytest.raises(TypeError):
        scan_root(file_path)
