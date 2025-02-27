"""
Seeing if the model that I make can recapitulate schmuker results.
"""

import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from model import molecule_model as mm


def fix_schmuker(dataset):
    """
    canonical smmiles and remove nas
    """
    df = pd.read_csv(dataset)
    reformated_df = pd.DataFrame(
        columns=['Protein', 'odorant', 'SMILES', 'output', 'Protein sequence']
    )

    # make cannonical SMILES
    for i, row in df.iterrows():
        # ignoring nans
        if pd.isna(row['output']):
            continue
        
        else:
            reformated_df = pd.concat(
                [
                    reformated_df,
                    pd.DataFrame({
                        'Protein': row['Protein'],
                        'odorant': row['odorant'],
                        'SMILES': Chem.MolToSmiles(Chem.MolFromSmiles(row['SMILES'])),
                        'output': row['output'],
                        'Protein sequence': row['Protein sequence']
                    }, index=[i])
                ],
                ignore_index=True
            )

    return reformated_df


def main(dataset):
    # make the train and test smiles cannonical
    test_df = fix_schmuker(dataset)
    train_df = fix_schmuker(dataset)

    # predict with all embeddings
    emb_types = mm.get_embedding_types()
    for emb_type in tqdm(emb_types):
        embs = mm.generate_embeddings(dataset, emb_type)
        print('here')


if __name__ == '__main__':
    dataset = 'data/Schmuker/test.csv'
    main(dataset)
