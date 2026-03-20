from typing import cast

import awkward as ak
import polars as pl
import uproot
from uproot.behaviors.TBranch import HasBranches
from uproot.behaviors.RNTuple import HasFields
from polars.io.plugins import register_io_source


def scan_root(file_name: str, tree_name: str | None = None) -> pl.LazyFrame:
    tree = uproot.open(file_name)
    if tree_name is not None:
        tree = tree[tree_name]
    if not isinstance(tree, (HasBranches, HasFields)):
        raise Exception(f"{file_name} does not contain a TTree or RNTuple named {tree_name}")

    # Create empty DataFrame to detect schema
    schema_df = pl.from_arrow(ak.to_arrow_table(tree.arrays(entry_stop=0), extensionarray=False))
    schema_df = cast(pl.DataFrame, schema_df)
    schema = schema_df.schema

    is_rntuple = isinstance(tree, HasFields)

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | str | None,
    ):
        if batch_size is None:
            batch_size = "100 MB"

        if is_rntuple:
            # RNTuple: `expressions` is not yet implemented, use filter_name
            kwargs = {"step_size": batch_size, "entry_stop": n_rows}
            if with_columns is not None:
                kwargs["filter_name"] = with_columns
            batches = tree.iterate(**kwargs)
        else:
            batches = tree.iterate(expressions=with_columns, step_size=batch_size, entry_stop=n_rows)

        for batch in batches:
            df = pl.from_arrow(ak.to_arrow_table(batch, extensionarray=False))
            df = cast(pl.DataFrame, df)

            if predicate is not None:
                df = df.filter(predicate)

            yield df

    return register_io_source(io_source=source_generator, schema=schema)


def read_root(file_name: str, tree_name: str | None = None) -> pl.DataFrame:
    return scan_root(file_name, tree_name=tree_name).collect()
