"""
This script takes out receptors that are a problem and makes a new dataframe
with only receptors that don't have large overfitting problems
"""

import numpy as np
import pandas as pd

df = pd.read_csv("results/Davis/molecule_emb_only_w_prob_receptors.csv")

# indicies with receptors with over 10 r2 value
big_r2_idx = df.index[np.abs(df['shuf_mean']) > 10].to_list()
prob_receptors = list(set(list(df['receptor'][big_r2_idx])))


print(len(prob_receptors))
