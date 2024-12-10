"""
Combination model of molecules and proteins for prediction. A linear regression
with a protein embeddings concatenated to a molecule embedding.

TODO:
 - CDhit proteins / scffold split molecules
"""

import os
import numpy as np
import pandas as pd
import pickle as pkl
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
    parent_dir = os.path.dirname(dataset)
    save_path = os.path.join(parent_dir, 'featurized_proteins', 'prots.pkl')
    if os.path.isfile(save_path):
        return pkl.load(open(save_path, 'rb'))

    # get all proteins
    prots = []
    for i, row in pd.read_csv(dataset).iterrows():
        prots.append((row['Protein sequence'][:5], row['Protein sequence']))

    prots = list(set(prots))

    # load esm model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # not prosmith
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # generated protein embeddings
    _, _, batch_tokens = batch_converter(prots)
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

    # putting representations into a dict and saving
    feat_dict = {}
    for p, rep in zip(prots, sequence_representations):
        feat_dict[p[1]] = rep

    pkl.dump(feat_dict, open(save_path, "wb"))

    return feat_dict


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
    print('done')
