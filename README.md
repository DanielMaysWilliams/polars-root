# Polars ROOT #

[Polars](https://pola.rs/) plugin for reading [CERN's ROOT file format](https://root.cern/).
Uses [Uproot](https://github.com/scikit-hep/uproot5) and [Awkward Arrays](https://github.com/scikit-hep/awkward) under the hood.

## Installation ##

`polars-root` supports Python 3.10 and later.
To install for use with standard `polars`, run:

```bash
pip install 'polars-root[standard]'
```

To install for use with `polars-u64-idx`, run:

```bash
pip install 'polars-root[u64-idx]'
```

## Usage ##
```python
import polars as pl
import polars_root as pr

# Read TTree `tree_name` from a ROOT file into a Polars DataFrame
df = pr.read_root("example.root", "tree_name")

# Read TTree `tree_name` from a ROOT file into a Polars LazyFrame
lf = pr.scan_root("example.root", "tree_name")

# Load TTree `tree_name` from a ROOT file into a LazyFrame, perform some operations, and then sink to parquet
(
    pr.scan_root("example.root", "tree_name")
    .filter(pl.col("some_column") > 0)
    .select(["some_column", "another_column"])
    .sink_parquet("output.parquet")
)

# Also supports opening a ROOT file and jumping straight to a TTree using a colon, as in Uproot:
df = pr.read_root("example.root:tree_name")
```
