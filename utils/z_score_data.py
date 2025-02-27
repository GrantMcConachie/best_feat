"""
Z-scores a given dataset
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore

dataset = "data/CareyCarlson/CC_reformat.csv"

df = pd.read_csv(dataset)
print(df)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].apply(zscore)

df.to_csv("data/CareyCarlson/CC_reformat_z.csv", index=False)
