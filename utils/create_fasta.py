"""
Creates fasta files for the CDhit algorithm
"""

import os
import pandas as pd


def main(dataset):
    # read csv
    df = pd.read_csv(dataset)
    prots = list(df['Protein sequence'].unique())

    # make fasta file
    save_path = os.path.dirname(dataset) + "/cdhit/cd.fasta"
    with open(save_path, 'w') as f:
        for i, row in enumerate(prots):
            f.write(f'>seq{i}\n{row}\n')


if __name__ == '__main__':
    dataset = 'data/CareyCarlson/CC_reformat_z.csv'
    main(dataset)
