import pandas as pd
from dashboards.base import Dashboard, registry, Slice


def slice_by_depth(df: pd.DataFrame) -> list[Slice]:
    dfs = df.groupby("depth")

    return [
        Slice(
            name=f"Depth {depth}", 
            df=df,
            metrics={
                "generate_codehop/score": df["score"].mean(),
            }
        )
        for depth, df in dfs
    ]

def slice_by_file(df: pd.DataFrame) -> list[Slice]:
    dfs = df.groupby("file_name")
    return [
        Slice(
            name=f"File {file}", 
            df=df,
            metrics={
                "generate_codehop/score": df["score"].mean(),  
            }
        )
        for file, df in dfs
    ]

registry.register(Dashboard(
    name="CodeHop",
    filters={"$and": [{"tags": "codehop"}, {"tags": "train"}]},
    table="generate_codehop/table",
    score_metric="generate_codehop/score",
    step="train/optimizer_step",
    slice_fns=[
        slice_by_depth,
        slice_by_file,
    ]
))
    