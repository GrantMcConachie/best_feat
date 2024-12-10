"""
Combination model of molecules and proteins for prediction. A linear regression
with a protein embeddings concatenated to a molecule embedding.

TODO:
 - CDhit proteins / scffold split molecules
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

import esm
import torch

import scripts.best_feat_regression as bfr


def generate_embeddings(dataset, mol_emb_type):
    """
    generates molecular and protein embeddings
    """
    # molecular embeddings
    mol_embs = bfr.generate_embeddings(dataset, mol_emb_type)

    # protein embeddings

    # check if saved

    # get all proteins
    prots = []
    for i, row in pd.read_csv(dataset).iterrows():
        prots.append((row['Protein'], row['Protein sequence']))

    prots = list(set(prots))

    # load esm model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # not prosmith
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # generated protein embeddings
    batch_labels, batch_strs, batch_tokens = batch_converter(prots)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(
            token_representations[i, 1:tokens_len - 1].mean(0)
        )


def main(dataset):
    emb_types_mol = bfr.get_embedding_types()
    for dataset in datasets:
        emb_scores = []
        for emb_type in tqdm(emb_types_mol):
            embs = generate_embeddings(dataset, emb_type)
            break
        break


if __name__ == '__main__':
    datasets = [
        'data/HallemCarlson/hc_with_prot_seq.csv',
        # 'data/Davis/davis.csv'
    ]
    main(datasets)
