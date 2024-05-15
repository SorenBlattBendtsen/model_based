import pandas as pd
import numpy as np

df = pd.read_csv("data/2023/nordic_energy_data.csv")

def transpose_for_country_code(df, country_code):
    """"
    Function to transpose the data for a specific country code.
    Data for other countries will be included in the same dataframe as additional columns.
    Input:
    - df: pandas dataframe
    - country_code: string, the country code to transpose the data for, eg. 'DK_1'
    Output:
    - df_"country_code": pandas dataframe, the data transposed for the specific country code
    """
    # drop columns that includes "Ex" and "Imp", as well as "Actual Load"
    df = df.loc[:,(~df.columns.str.contains("Ex") & ~df.columns.str.contains("Imp"))]
    df = df.drop(columns=["Unnamed: 0", "Actual Load"])
    
    # split the data into two dataframes, one for the country and one for the rest
    df_country = df[df['country_code'] == country_code]
    df_others = df[df['country_code'] != country_code]
    df_others = df_others.drop(columns=['DA-price [EUR/MWh]'])

    # pivot the data for the country
    df_others = df_others.pivot(index='Timestamp', columns='country_code').reset_index()
    # make only one level of columns, adding country_code as suffix
    df_others.columns = ['_'.join(col).strip() for col in df_others.columns.values]
    # rename timestamp
    df_others.rename(columns={'Timestamp_':'Timestamp'}, inplace=True)
    # merge the two dataframes
    df_country = pd.merge(df_country, df_others, on='Timestamp', how='left')
    # Make NaN values 0
    df_country = df_country.fillna(0)
    # drop columns that are all 0
    df_country = df_country.loc[:, (df_country != 0).any(axis=0)]

    return df_country

# Example for DK_1
df_dk1 = transpose_for_country_code(df, 'DK_1')

# normalize data with mean = 0
def normalize(df):
    """
    Function to normalize the data.
    Input:
    - df: pandas dataframe
    Output:
    - df: pandas dataframe, the normalized data
    """
    df = df.set_index('Timestamp')
    df = (df - df.mean())/df.std()
    df = df.reset_index()
    return df