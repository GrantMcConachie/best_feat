"""
Finding the best molecular featurizer for M2OR (a classification task)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import deepchem as dc

import best_feat_regression as r


def run_lr(dataset, embs):
    # init
    randsplitter = dc.splits.RandomStratifiedSplitter()  # NOTE: random instead
    df = pd.read_csv(dataset)
    proteins = df['Protein'].unique()

    # loop through unique proteins
    scores_per_protein = []
    for protein in proteins:
        skip_prot = False
        smiles = df['SMILES'][df['Protein'] == protein]
        output = np.array(df['output'][df['Protein'] == protein])
        embeddings = np.array([embs[i] for i in smiles])

        # ignore proteins with >10 values
        if len(smiles) < 50:
            continue

        # scaffold split molecules
        dataset_ = dc.data.NumpyDataset(
            X=output,  # NOTE: to get the splitter to work effectivly
            y=embeddings,
            ids=smiles
        )
        train, val, test = randsplitter.train_valid_test_split(
            dataset_,
            seed=1245
        )

        # if all ones or zeros, skip
        for set in [train, val, test]:
            if np.all(set.X == 1) or np.all(set.X == 0):
                skip_prot = True

        if skip_prot:
            continue

        # specify train and val for hyperparameter sweep
        train_val_x = np.concatenate([train.y, val.y])
        train_val_y = np.concatenate([train.X, val.X])
        test_fold = np.concatenate(
            [
                -1 * np.ones(shape=len(train.y)),
                np.zeros(shape=len(val.y))
            ]
        )

        # defining regressor
        reg = LogisticRegression()
        param_grid = {
            'C': np.logspace(-10, 10, num=21)
        }

        # hyperparameter search
        clf = GridSearchCV(
            reg,
            param_grid=param_grid,
            cv=PredefinedSplit(test_fold)
        )
        clf.fit(train_val_x, train_val_y)

        # evaluating metrics
        best_loc = np.where(clf.cv_results_['rank_test_score'] == 1)[0][0]
        best_C = list(clf.cv_results_['params'][best_loc].values())[0]
        reg.C = best_C
        reg.fit(X=train_val_x, y=train_val_y)

        # predicting test set and recording
        preds = reg.predict(test.y)
        score = matthews_corrcoef(test.X, preds)
        scores_per_protein.append(
            (
                protein,
                best_C,
                score
            )
        )

    return scores_per_protein


def main(datasets):
    emb_types = r.get_embedding_types()
    for dataset in datasets:
        emb_scores = []
        for emb_type in tqdm(emb_types):
            embs = r.generate_embeddings(dataset, emb_type)
            embs = r.clean_embeddings(embs)
            embs = r.reduce_embs(embs)
            scores = run_lr(dataset, embs)

            emb_scores.append((emb_type, scores))

        r.plot_results(emb_scores, dataset, 'mcc')

    plt.show()


if __name__ == '__main__':
    main(['data/M2OR/pairs_ec50.csv'])
