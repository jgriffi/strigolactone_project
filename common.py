import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors, Lipinski
from rdkit import Chem
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tqdm.notebook import tqdm
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu


def start_pipeline(df):
    return df.copy()   


def tidy(df):
    '''
    Drop nulls and duplicates.
    Split smiles on '.' and take the longest
    '''
    df.dropna(subset=['standard_value', 'canonical_smiles'], axis=0, inplace=True)
    df.drop_duplicates(['canonical_smiles'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    cmpd = [str(i).split('.') for i in df['canonical_smiles']]
    cleaned_canonical_smiles = [max(i, key=len) for i in cmpd]
    
    df.drop(columns='canonical_smiles', axis=1, inplace=True)
    
    df['cleaned_canonical_smiles'] = cleaned_canonical_smiles
    
    return df


def select_cols(df):
    cols = ['molecule_chembl_id', 'cleaned_canonical_smiles', 'standard_value']
    df = df[cols]
    df['standard_value'] = df['standard_value'].astype(float)
    return df


def add_bioactivity(df):
    '''
    Labeling compounds as either being active, inactive, or intermediate.
    The standard value is reported as IC50.
    Compounds having values of less than 1,000 nM will be considered 'active'
    while those greater than 10,000 nM will be considered 'inactive'
    and those values in between 1,000 and 10,000 nM, 'intermediate'.
    '''
    
    df['bioactivity_class'] = ['inactive' if i >= 10_000
                               else 'active' if i <= 1_000
                               else 'intermediate'
                               for i in df['standard_value']]
    
    return df


def lipinski(df):
    '''
    Christopher Lipinski, a scientist at Pfizer,came up with a set of rule-of-thumb for evaluating
    the druglikeness of compounds. Such druglikeness is based on the Absorption,
    Distribution, Metabolism and Excretion (ADME) that is also known as the pharmacokinetic profile.
    Lipinski analyzed all orally active FDA-approved drugs in the formulation of what is to be
    known as the Rule-of-Five or Lipinski's Rule.
    
    The Lipinski's Rule stated the following:
        - Molecular weight < 500 Dalton
        - Octanol-water partition coefficient (LogP) < 5
        - Hydrogen bond donors < 5
        - Hydrogen bond acceptors < 10
    '''
    
    moldata = [Chem.MolFromSmiles(elem) for elem in df['cleaned_canonical_smiles']]
    
    desc_MolWt = [Descriptors.MolWt(mol) for mol in moldata]
    desc_MolLogP = [Descriptors.MolLogP(mol) for mol in moldata]
    desc_NumHDonors = [Lipinski.NumHDonors(mol) for mol in moldata]
    desc_NumHAcceptors = [Lipinski.NumHAcceptors(mol) for mol in moldata]
      
    descriptors = pd.DataFrame(list(zip(desc_MolWt,
                                        desc_MolLogP,
                                        desc_NumHDonors,
                                        desc_NumHAcceptors
                                       )), columns=["MW","LogP","NumHDonors","NumHAcceptors"]
                              ).astype(float)
    df = pd.concat([df, descriptors], axis=1)
    
    return df


def normalize_standard_value(df):
    '''
    Normalize standard values by setting upper limit of 1^8
    ''' 
    
    norm = [i if i <= 100_000_000 else 100_000_000 for i in df["standard_value"]]
    # create norm column
    df["standard_value_norm"] = norm
    # drop standard_value column
    df.drop(labels="standard_value", axis=1, inplace=True)
    
    return df


def pIC50(df):
    '''
    Convert IC50 to pIC50:
    To allow IC50 data to be more uniformly distributed, 
    we will convert IC50 to the negative logarithmic scale 
    which is essentially -log10(IC50).

    This function will accept a DataFrame as input and will:
        - Take the IC50 values from the standard_value column 
          and convert them from nM to M by multiplying the value by 10^-9
        - Take the molar value and apply log10(IC50)
        - Delete the standard_value column and create a new pIC50 column
    '''
    molar = 1e-9
    pIC50 = [-np.log10(i * molar) for i in df["standard_value_norm"]]
    # add pIC50 column
    df["pIC50"] = pIC50
    # drop standard_value_norm
    df.drop(labels="standard_value_norm", axis=1, inplace=True)
    
    return df


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = Path.joinpath(IMAGES_PATH, f"{fig_id}.{fig_extension}")
    print("Saving figure", fig_id)
    print(f"Image saved at - {path}")
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    

def VarianceThreshold_selector(data, threshold=(.8 * (1 - .8))):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit_transform(data)
    return(data[data.columns[selector.get_support(indices=True)]])


def prepare_dataset(df, descriptors):
    '''
    - Prepare dataset for modelling by merging the cleaned strigolactone
      datframe with the descriptors.
    - select the X matrix and remove low variance
    - select the y matrix
    '''
    
    df = df.merge(descriptors.rename(columns={'Name': 'molecule_chembl_id'}),
             on='molecule_chembl_id',how='left')
    df.dropna(axis=0, inplace=True)
    
    X = df.drop(columns=['molecule_chembl_id',
                         'cleaned_canonical_smiles',
                         'bioactivity_class',
                         'MW',
                         'LogP',
                         'NumHDonors',
                         'NumHAcceptors',
                         'pIC50'], axis=1)
    
#     selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
#     X = selection.fit_transform(X)

    X = VarianceThreshold_selector(X)
    
    y = df['pIC50']
    
    # Create training and test sets. 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardise the features
    scaler = StandardScaler()

    # fit_transform train
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    # Transform test
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test, df, X


def display_scores(scores):
    print('scores:', scores)
    print('mean:', scores.mean())
    print('standard deviation:', scores.std())
    

def mannwhitney(df, descriptor='pIC50', col='bioactivity_class'):
    # https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
    
    
    # select active and inactive
    active = [x for x,y in zip(df[descriptor], df[col]) if y == 'active']
    inactive = [x for x,y in zip(df[descriptor], df[col]) if y == 'inactive']
    
    # compare samples
    stat, p = mannwhitneyu(active, inactive)
    
    # interpret
    alpha = 0.05
    if p > alpha:
        interpretation = 'Same distribution (fail to reject H0)'
    else:
        interpretation = 'Different distribution (reject H0)'
    
    results = pd.DataFrame({'Descriptor': descriptor,
                            'Statistics': stat,
                            'p': p,
                            'alpha': alpha,
                            'Interpretation': interpretation
                           }, index=[0])
    
    return results