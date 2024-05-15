import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
    # Drop columns where the value for all rows are the same
    df_country = df_country.loc[:, df_country.nunique() != 1]

    return df_country


def split_and_normalize(df):

    # Make timestamp the index
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.set_index("Timestamp", inplace=True)

    # split the data into features and target
    y = df["DA-price [EUR/MWh]"]
    X = df.drop(columns=["DA-price [EUR/MWh]"])

    # split the data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Normalize the data based on the training set
    x_train_mean = X_train.mean()
    x_train_std = X_train.std()

    X_train = (X_train - x_train_mean) / x_train_std
    X_test = (X_test - x_train_mean) / x_train_std

    return X_train, X_test, y_train, y_test