"""
Benchmarks all featurizers on molfeat against OR datasets of interest.
"""

import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from scipy.stats import zscore
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ShuffleSplit

from molfeat.trans import MoleculeTransformer
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.pretrained import PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer

import deepchem as dc


def get_embedding_types():
    """
    Returns all embedding types in molfeat
    """
    transformers = [
        ('trans', 'Roberta-Zinc480M-102M'),
        ('trans', 'GPT2-Zinc480M-87M'),
        ('trans', 'ChemGPT-19M'),
        ('trans', 'MolT5'),
        ('trans', 'ChemBERTa-77M-MTR'),
    ]
    gnns = [
        ('graph', 'gin_supervised_infomax'),
        ('graph', 'gin_supervised_edgepred'),
        ('graph', 'gin_supervised_contextpred')
    ]
    basic_fingerprints = [
        ('base', 'CATS'),
        ('base', 'MordredDescriptors'),
        ('base', 'Pharmacophore2D'),
        ('base', 'RDKitDescriptors2D'),
        ('base', 'ScaffoldKeyCalculator')
    ]
    ecfp = [('ecfp', 'ecfp')]

    return basic_fingerprints + ecfp + transformers + gnns


def generate_embeddings(dataset, emb_type):
    """
    Generates embeddings for all smiles in every dataset
    """
    # check if saved version
    emb_class = emb_type[0]
    emb_name = emb_type[1]
    parent_dir = os.path.dirname(dataset)
    save_path = os.path.join(parent_dir, 'featurized_mols', f'{emb_name}.pkl')
    if os.path.isfile(save_path):
        return pkl.load(open(save_path, 'rb'))

    # get all smiles
    smiles = list(pd.read_csv(dataset)['SMILES'])
    smiles = np.array(list(set(smiles)))

    # featurize
    if emb_class == 'base':
        featurizer = MoleculeTransformer(emb_name, dtype=np.float64)
        feats = featurizer(smiles)

    elif emb_class == 'ecfp':
        featurizer = FPVecTransformer(kind='ecfp', dtype=float)
        feats = featurizer(smiles)

    elif emb_class == 'trans':  # pretrained models
        featurizer = PretrainedHFTransformer(
            kind=emb_name,
            notation='smiles',
            dtype=float
        )
        feats = featurizer(smiles)

    elif emb_class == 'graph':
        featurizer = PretrainedDGLTransformer(
            kind=emb_name,
            dtype=float
        )
        feats = featurizer(smiles)

    # save
    feat_dict = {}
    for smi, feat in zip(smiles, feats):
        feat_dict[smi] = feat

    pkl.dump(feat_dict, open(save_path, "wb"))

    return feat_dict


def clean_embeddings(embs):
    """
    Removes NaNs from all the embeddigns if there is any. This is necessary
    for mordred descriptors.
    """
    values = np.array(list(embs.values()))
    nans = np.unique(np.where(np.isnan(values))[1])
    cleaned_embs = {}
    for key, value in embs.items():
        cleaned_embs[key] = np.delete(value, nans)

    return cleaned_embs


def reduce_embs(emb_dict):
    """
    Does PCA on embeddings and reduces the dimensionality to explain 99% of the
    variance
    """
    embs = np.array(list(emb_dict.values()))
    reduced = PCA(n_components=0.99, svd_solver='full').fit_transform(embs)
    reduced_embs = {}
    for i, (key, _) in enumerate(emb_dict.items()):
        reduced_embs[key] = reduced[i]

    return reduced_embs


def calc_rmse(preds, true):
    """
    calculated the root mean squared error of predicted and true values
    """
    return np.sqrt(1 / len(preds) * np.sum((preds - true) ** 2))


def run_lr(dataset, embs, regressor='r'):
    """
    Runs a 5-fold linear regression for a given embedding and dataset
    """
    # init
    seed = 12345
    scaffoldsplitter = dc.splits.ScaffoldSplitter()
    rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    df = pd.read_csv(dataset)
    proteins = df['Protein'].unique()

    # loop through unique proteins
    scores_per_protein = []
    for protein in proteins:
        smiles = df['SMILES'][df['Protein'] == protein]
        output = zscore(np.array(df['output'][df['Protein'] == protein]))
        embeddings = np.array([embs[i] for i in smiles])

        # defining regressor
        if regressor == 'r':
            reg = Ridge()
        elif regressor == 'l':
            reg = Lasso()

        # parameters to sweep over
        param_grid = {
            'alpha': np.logspace(-10, 10, num=21)
        }

        # 5-fold cross val for random shuffling
        random_shuf_scores = []
        for i, (train_index, test_index) in enumerate(rs.split(output)):
            clf = GridSearchCV(
                reg,
                param_grid=param_grid,
            )
            clf.fit(embeddings[train_index], output[train_index])
            best_model = clf.best_estimator_
            best_model.fit(embeddings[train_index], output[train_index])
            r2 = best_model.score(embeddings[test_index], output[test_index])
            random_shuf_scores.append(r2)

        # scaffold split molecules
        dataset_ = dc.data.NumpyDataset(
            X=embeddings,
            y=output,
            ids=smiles
        )
        train, val, test = scaffoldsplitter.train_valid_test_split(
            dataset_,
            seed=seed
        )

        # specify train and val for hyperparameter sweep
        train_val_x = np.concatenate([train.X, val.X])
        train_val_y = np.concatenate([train.y, val.y])
        test_fold = np.concatenate(
            [
                -1 * np.ones(shape=len(train.y)),
                np.zeros(shape=len(val.y))
            ]
        )

        # hyperparameter search
        clf = GridSearchCV(
            reg,
            param_grid=param_grid,
            cv=PredefinedSplit(test_fold)
        )
        clf.fit(train_val_x, train_val_y)

        # evaluating metrics
        clf.best_estimator_.fit(X=train_val_x, y=train_val_y)

        # predicting test set and recording
        scaf_score = clf.best_estimator_.score(test.X, test.y)
        scores_per_protein.append(
            (
                protein,
                random_shuf_scores,
                scaf_score
            )
        )

    return scores_per_protein


def plot_results(results, dataset, ylabel='r2'):
    """
    plots embedding scores on a given dataset
    """
    # getting average score over all proteins
    emb_names_NN = []
    emb_scores_NN_shuf = []
    emb_scores_NN_scaf = []
    emb_names_P = []
    emb_scores_P_shuf = []
    emb_scores_P_scaf = []

    for result in results:
        shuf_protein_scores = np.array([i[1] for i in result[1]]).flatten()
        scaf_protein_scores = np.array([i[2] for i in result[1]])
        if result[0][0] == 'graph' or result[0][0] == 'trans':
            emb_names_NN.append(result[0][1])
            emb_scores_NN_shuf.append(shuf_protein_scores)
            emb_scores_NN_scaf.append(scaf_protein_scores)

        else:
            emb_names_P.append(result[0][1])
            emb_scores_P_shuf.append(shuf_protein_scores)
            emb_scores_P_scaf.append(scaf_protein_scores)

    # plotting
    fig, axs = plt.subplots(1, 2)

    # physicochemical descriptors
    axs[0].violinplot(
        np.array(emb_scores_P_shuf).T,
        showmeans=True,
        positions=np.arange(len(emb_names_P))
        )
    for i, row in enumerate(emb_scores_P_shuf):
        axs[0].scatter(
            np.ones_like(row) * i, row, color="#cc6666", alpha=0.3, s=5
        )
    # axs[0].violinplot(emb_scores_P_scaf)
    axs[0].set_title('Physicochemical Descriptors')
    axs[0].set_ylabel(ylabel)
    axs[0].set_ylim([-3, 1])
    axs[0].set_xticks(range(len(emb_names_P)))
    axs[0].set_xticklabels(emb_names_P)

    # Neural network embeddings
    axs[1].violinplot(
        np.array(emb_scores_NN_shuf).T,
        showmeans=True,
        positions=np.arange(len(emb_names_NN))
    )
    for i, row in enumerate(emb_scores_NN_shuf):
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


def main(datasets):
    emb_types = get_embedding_types()
    for dataset in datasets:
        emb_scores = []
        for emb_type in tqdm(emb_types):
            embs = generate_embeddings(dataset, emb_type)
            embs = clean_embeddings(embs)
            embs = reduce_embs(embs)
            scores = run_lr(dataset, embs, regressor='r')

            emb_scores.append((emb_type, scores))

        plot_results(emb_scores, dataset)

    plt.show()


if __name__ == '__main__':
    datasets = [
        'data/HallemCarlson/hc_data_reformat.csv',
        'data/Davis/davis.csv'
    ]
    main(datasets)
