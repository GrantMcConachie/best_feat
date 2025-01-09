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
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import ShuffleSplit, GridSearchCV

import esm
import torch

import scripts.best_feat_regression as r


def generate_embeddings(dataset, mol_emb_type):
    """
    generates molecular and protein embeddings
    """
    # molecular embeddings
    mol_embs = r.generate_embeddings(dataset, mol_emb_type)

    # protein embeddings
    # check if saved
    parent_dir = os.path.dirname(dataset)
    save_path = os.path.join(parent_dir, 'featurized_proteins', 'prots.pkl')
    if os.path.isfile(save_path):
        return (mol_embs, pkl.load(open(save_path, 'rb')))

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

    # chunking into size 10 batches for cpu
    split_batch_tokens = torch.tensor_split(
        batch_tokens,
        len(batch_tokens) // 10,
        dim=0
    )
    split_batch_lens = torch.tensor_split(
        batch_lens,
        len(batch_tokens) // 10,
    )

    # Extract per-residue representations
    for (tok, lens) in zip(split_batch_tokens, split_batch_lens):
        with torch.no_grad():
            results = model(
                tok,
                repr_layers=[33],
                return_contacts=True
            )
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

    return (mol_embs, feat_dict)


def run_regression(dataset, mol_emb, prot_emb, regressor='r'):
    """
    Does 5 fold cross validation for different molecular embeddings with
    appended esm embeddings added
    """
    # init
    seed = 12345

    # getting data
    df = pd.read_csv(dataset)
    x, y = [], []
    for _, row in df.iterrows():
        mol_feat = mol_emb[row['SMILES']]
        prot_feat = prot_emb[row['Protein sequence']]
        combo = np.concatenate([mol_feat, prot_feat])
        x.append(combo)
        y.append(row['output'])

    # convert to numpy
    x = np.array(x)
    y = zscore(np.array(y))

    # defining regressor
    if regressor == 'r':
        reg = Ridge()
    elif regressor == 'l':
        reg = Lasso()

    # parameters to sweep over
    param_grid = {
        'alpha': np.logspace(-10, 10, num=21)
    }

    # defining splits of the data
    rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)

    # performing cross val
    scores = []
    for train_index, test_index in rs.split(y):
        clf = GridSearchCV(
            reg,
            param_grid=param_grid,
        )
        clf.fit(x[train_index], y[train_index])
        best_model = clf.best_estimator_
        best_model.fit(x[train_index], y[train_index])
        r2 = best_model.score(x[test_index], y[test_index])
        scores.append(r2)

    return scores


def plot_results(emb_scores, dataset, ylabel='r2'):
    scores_NN = []
    scores_P = []
    emb_names_NN = []
    emb_names_P = []

    for i in emb_scores:
        if i[0][0] == 'graph' or i[0][0] == 'trans':
            scores_NN.append(i[1])
            emb_names_NN.append(i[0][1])
            print(f'{i[0][1]}: mean {np.mean(i[1])} std: {np.std(i[1])}')

        else:
            scores_P.append(i[1])
            emb_names_P.append(i[0][1])
            print(f'{i[0][1]}: mean {np.mean(i[1])} std: {np.std(i[1])}')

    # plotting
    fig, axs = plt.subplots(1, 2)

    # physicochemical descriptors
    axs[0].violinplot(
        np.array(scores_P).T,
        showmeans=True,
        positions=np.arange(len(emb_names_P))
        )
    for i, row in enumerate(scores_P):
        axs[0].scatter(
            np.ones_like(row) * i, row, color="#cc6666", alpha=0.7, s=5
        )
    axs[0].set_title('Physicochemical Descriptors')
    axs[0].set_ylabel(ylabel)
    axs[0].set_ylim([-3, 1])
    axs[0].set_xticks(range(len(emb_names_P)))
    axs[0].set_xticklabels(emb_names_P)

    # Neural network embeddings
    axs[1].violinplot(
        np.array(scores_NN).T,
        showmeans=True,
        positions=np.arange(len(emb_names_NN))
    )
    for i, row in enumerate(scores_NN):
        axs[1].scatter(
            np.ones_like(row) * i, row, color="#cc6666", alpha=0.3, s=5
        )
    # axs[1].violinplot(emb_scores_NN_scaf)
    axs[1].set_title('Neural Network Embeddings')
    axs[1].set_ylabel(ylabel)
    axs[1].set_ylim([-3, 1])
    axs[1].set_xticks(range(len(emb_names_NN)))
    axs[1].set_xticklabels(emb_names_NN)

    fig.suptitle(dataset)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()


def main(dataset):
    emb_types_mol = r.get_embedding_types()
    for dataset in datasets:
        emb_scores = []
        for emb_type in tqdm(emb_types_mol):
            mol_emb, prot_emb = generate_embeddings(dataset, emb_type)
            mol_emb = r.clean_embeddings(mol_emb)
            mol_emb = r.reduce_embs(mol_emb)  # idk about this
            scores = run_regression(dataset, mol_emb, prot_emb, regressor='r')
            emb_scores.append((emb_type, scores))
        plot_results(emb_scores, dataset)
    plt.show()


if __name__ == '__main__':
    datasets = [
        'data/Davis/davis.csv',
        'data/HallemCarlson/hc_with_prot_seq_z.csv',
    ]
    main(datasets)
