"""
Splits data up by molecule and only uses protein information to do prediction.
"""

import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from scipy.stats import zscore
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold, StratifiedShuffleSplit

from molfeat.trans import MoleculeTransformer
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans.pretrained import PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer

import deepchem as dc

import model.combo_model as c
import model.molecule_model as m


def run_lr(dataset, embs, regressor='r'):
    """
    Runs a 5-fold linear regression for a given embedding and dataset
    """
    # init
    seed = 12345
    scaffoldsplitter = dc.splits.ScaffoldSplitter()
    ss = KFold(n_splits=5, shuffle=False)
    rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    df = pd.read_csv(dataset)
    smiles = df['SMILES'].unique()

    # loop through unique smiles
    scores_per_smi = []
    for smi in smiles:
        protein = df['Protein sequence'][df['SMILES'] == smi]
        output = np.array(df['output'][df['SMILES'] == smi])
        embeddings = np.array([embs[i] for i in protein])

        # parameters to sweep over
        param_grid = {
            'alpha': np.logspace(-10, 10, num=21)
        }

        # defining regressor
        if dataset == 'data/M2OR/pairs_ec50.csv':  # binary data
            reg = LogisticRegression()
            param_grid = {
                'C': np.logspace(-10, 10, num=21)
            }

            # ignore proteins with < 50 values
            if len(protein) < 23:
                continue

            # ignore output with only 1 positive example
            if len(np.where(output == 1)[0]) < 2 or len(np.where(output == 0)[0]) < 2:
                continue

            # have to have stratified splitting
            rs = StratifiedShuffleSplit(n_splits=5, random_state=seed)
            splits = rs.split(embeddings, output)

        elif regressor == 'r':
            reg = Ridge()
            splits = rs.split(output)

        elif regressor == 'l':
            reg = Lasso()
            splits = rs.split(output)

        # 5-fold cross val for random shuffling
        random_shuf_scores = []
        for i, (train_index, test_index) in enumerate(splits):
            clf = GridSearchCV(
                reg,
                param_grid=param_grid,
                n_jobs=-1
            )
            clf.fit(embeddings[train_index], output[train_index])
            best_model = clf.best_estimator_

            # change scoring based on dataset
            if dataset == 'data/M2OR/pairs_ec50.csv':
                preds = best_model.predict(embeddings[test_index])
                score = matthews_corrcoef(output[test_index], preds)
                random_shuf_scores.append(score)

            else:
                r2 = best_model.score(embeddings[test_index], output[test_index])
                random_shuf_scores.append(r2)

        scores_per_smi.append(
            (
                smi,
                random_shuf_scores,
            )
        )

    return scores_per_smi


def plot_results(results, datasets, regressor, ylabel='r2/mcc'):
    """
    plots embedding scores on a given dataset
    """
    # getting average score over all proteins
    dataset_names = [os.path.basename(i) for i in datasets]

    for i, (result, dataset) in enumerate(zip(results, datasets)):
        shuf_protein_scores = np.array([i[1] for i in result]).flatten()

        # ouput to terminal
        print(dataset)
        print('shuf mean:', np.mean(shuf_protein_scores))
        print('shuf std:', np.std(shuf_protein_scores))
        print('\n')

        # plotting
        fig, axs = plt.subplots()

        axs.hist(
            shuf_protein_scores,
            color='#CC6666',
            edgecolor='black',
            bins=100
        )
        axs.axvline(x=np.mean(shuf_protein_scores), color='#9FB798', lw=2, ls='--')
        axs.set_title(os.path.basename(dataset))
        axs.set_ylabel(ylabel)
        
        fig.autofmt_xdate(rotation=45)
        plt.tight_layout()


def tabulate(emb_scores, dataset, regressor):
    """
    Saves data to a csv
    """
    # create empty dataframe and a place for it to go
    path = os.path.dirname(dataset).replace("data/", "results/")
    path += "/molecule_emb_only.csv"
    df = pd.DataFrame(
        columns=[
            'embedding',
            'shuf_mean',
            'scaf_mean',
            'shuf1',
            'shuf2',
            'shuf3',
            'shuf4',
            'shuf5',
            'scaf1',
            'scaf2',
            'scaf3',
            'scaf4',
            'scaf5',
            'receptor',
            'regressor'
        ]
    )

    # loop through scores
    for score in emb_scores:
        emb = score[0][1]
        for prot in score[1]:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame({
                        'embedding': [emb],
                        'shuf_mean': [np.mean(prot[1])],
                        'scaf_mean': [np.mean(prot[2])],
                        'shuf1': [prot[1][0]],
                        'shuf2': [prot[1][1]],
                        'shuf3': [prot[1][2]],
                        'shuf4': [prot[1][3]],
                        'shuf5': [prot[1][4]],
                        'scaf1': [prot[2][0]],
                        'scaf2': [prot[2][1]],
                        'scaf3': [prot[2][2]],
                        'scaf4': [prot[2][3]],
                        'scaf5': [prot[2][4]],
                        'receptor': [prot[0]],
                        'regressor': regressor
                    })
                ],
                ignore_index=True
            )

    # save
    df.to_csv(path, index=False)


def main(datasets, regressor='r'):
    tot_scores = []
    for dataset in tqdm(datasets):
        _, embs = c.generate_embeddings(dataset) 

        # embs = r.reduce_embs(embs)
        tot_scores.append(run_lr(dataset, embs, regressor=regressor))

    plot_results(tot_scores, datasets, regressor)
    # tabulate(tot_scores, dataset, regressor)

    plt.show()


if __name__ == '__main__':
    datasets = [
        'data/M2OR/pairs_ec50.csv',
        # 'data/Davis/davis_z.csv',
        'data/HallemCarlson/hc_with_prot_seq_z.csv',
        'data/CareyCarlson/CC_reformat_z.csv'
    ]
    main(datasets)
