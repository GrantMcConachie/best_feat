from rdkit import Chem
import pandas as pd

# load data
df = pd.read_csv("data/HallemCarlson/hc_data.csv")
reformated_df = pd.DataFrame(
    columns=['RECEPTOR', 'odorant', 'SMILES', 'output']
)
spont_fr = df.loc[110]

# loop through rows
for i, (_, row) in enumerate(df.iterrows()):
    # skip spontaneous firing rates
    if row['odorant'] == 'spontaneous firing rate':
        continue

    # canonical smiles
    smiles = row['smiles']
    canon_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    # adjust for spontaneous firing rate
    net_fr = row.iloc[3:] - spont_fr.iloc[3:]

    # add to new csv
    for j, receptor in enumerate(row.keys()[3:]):
        reformated_df = pd.concat(
            [
                reformated_df,
                pd.DataFrame({
                    'RECEPTOR': receptor,
                    'odorant': row['odorant'],
                    'SMILES': canon_smiles,
                    'output': net_fr[receptor]
                }, index=[j + i*len(net_fr)])
            ],
            ignore_index=True
        )

# save csv
reformated_df.to_csv('data/HallemCarlson/hc_data_reformat.csv')
