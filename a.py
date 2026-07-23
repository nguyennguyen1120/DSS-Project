import pandas as pd
ctx = pd.read_parquet("data/processed/corpus.parquet")
print(ctx.columns.tolist())