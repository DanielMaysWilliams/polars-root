import pytest
import numpy as np
import uproot


@pytest.fixture(params=["ttree", "rntuple"])
def one_column_root_file(request, tmp_path):
    file_path = tmp_path / "test.root"
    data = {"x": np.array([1, 2, 3], dtype=np.int32)}
    with uproot.recreate(file_path) as f:
        if request.param == "ttree":
            f.mktree("tree", data)
        else:
            f.mkrntuple("tree", data)
    return str(file_path), "tree", data


@pytest.fixture(params=["ttree", "rntuple"])
def simple_root_file(request, tmp_path):
    file_path = tmp_path / "test.root"
    data = {
        "x": np.array([1, 2, 3], dtype=np.int32),
        "y": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        "z": np.array(["a", "b", "c"], dtype="U1"),
    }
    with uproot.recreate(file_path) as f:
        if request.param == "ttree":
            f.mktree("tree", data)
        else:
            f.mkrntuple("tree", data)
    return str(file_path), "tree", data
