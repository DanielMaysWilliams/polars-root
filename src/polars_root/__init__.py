import awkward as ak
import polars as pl
import uproot
from uproot.behaviors.TBranch import HasBranches
from polars.io.plugins import register_io_source

TYPENAME_MAPPING = {
    "int32_t": pl.Int32,
    "std::vector<int32_t>": pl.List(pl.Int32),
}


def root_reader(file_name: str, tree_name: str | None = None) -> pl.LazyFrame:
    tree = uproot.open(file_name)
    if tree_name is not None:
        tree = tree[tree_name]
    if not isinstance(tree, HasBranches):
        raise Exception(f"{file_name} does not contain a TTree named {tree_name}")

    # Create empty DataFrame from TTree to detect schema
    schema_df = pl.from_arrow(ak.to_arrow_table(tree.arrays(entry_stop=0), extensionarray=False))
    if isinstance(schema_df, pl.Series):
        schema_df = schema_df.to_frame()
    schema = schema_df.schema

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | str | None,
    ):
        if batch_size is None:
            batch_size = "100 MB"

        # Use built-in uproot batched iteration to yield DataFrames
        for batch in tree.iterate(expressions=with_columns, step_size=batch_size, entry_stop=n_rows):
            df = pl.from_arrow(ak.to_arrow_table(batch, extensionarray=False))
            if isinstance(df, pl.Series):
                df = df.to_frame()

            if predicate is not None:
                df = df.filter(predicate)

            yield df

    return register_io_source(io_source=source_generator, schema=schema)
