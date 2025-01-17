"""
This script takes out receptors that are a problem and makes a new dataframe
with only receptors that don't have large overfitting problems
"""

import numpy as np
import pandas as pd

df = pd.read_csv("results/Davis/molecule_emb_only_w_prob_receptors.csv")

# indicies with receptors with over 10 r2 value
big_r2_idx = df.index[
    (np.abs(df['shuf_mean']) > 10) | (np.abs(df['scaf_mean']) > 10)
].to_list()
prob_receptors = list(set(df['receptor'][big_r2_idx]))

# removing annoying receptors
for index, row in df.iterrows():
    if row['receptor'] in prob_receptors:
        df.drop(index, inplace=True)

# loop through embeddings
for emb in df['embedding'].unique():
    print(emb)
    print('shuf_mean:', df[df['embedding'] == emb]['shuf_mean'].mean())
    print('scaf_mean:', df[df['embedding'] == emb]['scaf_mean'].mean())
    print('\n')
