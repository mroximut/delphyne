import json

import pandas as pd

# pyright: basic
# fmt: on

"""
First, download the dataset from Hugging Face with the command:
`hf download allenai/ZebraLogicBench-private --repo-type=dataset`
Then, run this script to convert the Parquet files to CSV format.
"""

if __name__ == "__main__":
    df = pd.read_parquet(
        "examples/zebra_logic/datasets--allenai--ZebraLogicBench-private/snapshots/"
        "9f39ef490ae924437376657205025f26c0bd1af3/grid_mode/test-00000-of-00001"
        ".parquet"
    )

    df.to_csv(
        "examples/zebra_logic/datasets--allenai--ZebraLogicBench-private/"
        "zebra_logic_bench_puzzles.csv",
        index=False,
    )

    df_solutions = df.copy()
    df_solutions["solution"] = df_solutions["solution"].apply(
        lambda sol: json.dumps(
            [list(row) for row in sol["rows"]],
            ensure_ascii=False,
        )
    )
    df_solutions.drop(columns=["puzzle"], inplace=True)

    df_solutions.to_csv(
        "examples/zebra_logic/datasets--allenai--ZebraLogicBench-private/"
        "zebra_logic_bench_solutions.csv",
        index=False,
    )
