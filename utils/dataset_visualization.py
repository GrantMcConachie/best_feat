"""
Script to vizualize dataset statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def all_responses(dataset, data):
    """
    Plots every value in the dataset as a histogram
    """
    fig, axs = plt.subplots()
    axs.hist(data['output'], edgecolor='black', color='#cc6666')
    axs.set_xlabel('Response')
    axs.set_ylabel('count')
    axs.set_title(dataset)


def protein_tuning_curves(dataset, data, n_proteins=27):
    """
    Plots histograms of the responsivness of a protein to multiple odors

    Args:
        data (DataFrame) - data that is plotted
        n_proteins (int) - number of proteins to plot
    """
    # find unique proteins
    unique_proteins = list(set(data['Protein']))

    # histogram of their responses
    for i, prot in enumerate(unique_proteins):
        if i == n_proteins:
            break

        if i % 9 == 0:
            fig = plt.figure()
            fig.supxlabel('response')
            fig.supylabel('count')
            fig.suptitle(dataset)

        responses = data['output'][data['Protein'] == prot]
        ax = fig.add_subplot(3, 3, (i % 9) + 1)
        ax.hist(responses, edgecolor='k', color='#cc6666')
        if len(prot) > 5:
            ax.set_title(prot[:5])
        else:
            ax.set_title(prot)

        plt.tight_layout()


def calc_statistics(data):
    """
    Gettting some basic statistics from the data
    """
    all_responses = data['output']
    print('data mean:', np.mean(all_responses))
    print('data median:', np.median(all_responses))
    print('data varience:', np.var(all_responses))
    print('data min:', np.min(all_responses))
    print('data max:', np.max(all_responses))
    print('data count:', len(all_responses))


def main(dataset):
    df = pd.read_csv(dataset)
    all_responses(dataset, df)
    protein_tuning_curves(dataset, df)
    calc_statistics(df)
    plt.show()


if __name__ == '__main__':
    hc = 'data/HallemCarlson/hc_data_reformat.csv'
    davis = 'data/Davis/davis.csv'
    m2or = 'data/M2OR/pairs_ec50.csv'
    main(m2or)
