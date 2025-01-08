"""
Z-scores a given dataset
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore

dataset = "data/HallemCarlson/hc_with_prot_seq.csv"

df = pd.read_csv(dataset)
print(df)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].apply(zscore)

df.to_csv("data/HallemCarlson/hc_with_prot_seq_z.csv")
