#%%
import pandas as pd
from tqdm import tqdm
from entsoe import EntsoePandasClient
client = EntsoePandasClient(api_key='a6160036-4d49-4c39-960f-99c3c690b6da', retry_count=60, retry_delay=10)

def queryWeekAheadCapacities(mapping_table, start_date, end_date):

    """
    Query cross-border flows from the ENTSO-E API based on a mapping table.

    Parameters:
    - mapping_table (dict): A mapping table with key-value pairs representing the flow connections.
    - start_date (str): Start date for the query.
    - end_date (str): End date for the query.

    Returns:
    pd.DataFrame: DataFrame containing cross-border flow data with columns ['MTU', 'From', 'To', 'Flow_MW'].

    """

    df_list = []
    for key, values in tqdm(mapping_table.items(), desc='Processing NTCs'):
        for value in tqdm(values, desc=f'Processing NTC from {key}'):
            try:
                data_temp = client.query_net_transfer_capacity_weekahead(key, value, start=start_date, end=end_date)
                data_temp = data_temp.reset_index()
                data_temp['From'] = key
                data_temp['To'] = value
                df_list.append(data_temp)
            except:
                tqdm.write(f'No data for {key} -> {value}')

    print('Done')
    df_queried = pd.concat(df_list, ignore_index=True)
    df_queried = df_queried.rename(columns={0:'WeekAhead_NTC', 'index':'MTU'})

    return df_queried

def fromDailyToHourlyGranularity(NTCs, start_date, end_date):

    # Assuming 'index' is the index of your DataFrame
    # Convert 'index' to datetime if it's not already
    NTCs['MTU'] = pd.to_datetime(NTCs['MTU'])

    # Set 'index' as the index of your DataFrame
    NTCs.set_index('MTU', inplace=True)

    # Create a secondary index by combining 'From' and 'To'
    NTCs['secondary_index'] = NTCs['From'] + '_' + NTCs['To']

    NTCs_hourly = pd.DataFrame()
    for secondary in NTCs['secondary_index'].unique():
        selection = NTCs.loc[NTCs['secondary_index'] == secondary]

        new_index = pd.date_range(start=start_date, end=end_date, freq='h')
        selection = selection.reindex(new_index)

        # Reset the index to get 'dateTimeUtc' back as a column
        #selection.reset_index(inplace=True)

        # Forward fill to propagate values for new timestamps
        selection = selection.ffill()

        # concat the result to new_table
        NTCs_hourly = pd.concat([NTCs_hourly, selection])

    # drop the secondary index
    NTCs_hourly.drop('secondary_index', axis=1, inplace=True)

    # rename From and To to biddingZoneFrom and biddingZoneTo
    NTCs_hourly.rename(columns={'From':'biddingZoneFrom', 'To':'biddingZoneTo'}, inplace=True)
    NTCs_hourly.index.rename('MTU', inplace=True)

    return NTCs_hourly

def query_scheduled_exchanges_ENTSOE(mapping_table, start_date, end_date):

    """
    Query cross-border flows from the ENTSO-E API based on a mapping table.

    Parameters:
    - mapping_table (dict): A mapping table with key-value pairs representing the flow connections.
    - start_date (str): Start date for the query.
    - end_date (str): End date for the query.

    Returns:
    pd.DataFrame: DataFrame containing cross-border flow data with columns ['MTU', 'From', 'To', 'Flow_MW'].

    """

    df_list = []
    for key, values in tqdm(mapping_table.items(), desc='Processing scheduled exchanges'):
        for value in tqdm(values, desc=f'Processing scheduled exchanges from {key}'):
            try:
                data_temp = client.query_scheduled_exchanges(key, value, start=start_date, end=end_date, dayahead=True)
                data_temp = data_temp.reset_index()
                data_temp['From'] = key
                data_temp['To'] = value
                df_list.append(data_temp)
            except:
                tqdm.write(f'No data for {key} -> {value}')

    print('Done')
    df_queried = pd.concat(df_list, ignore_index=True)
    df_queried = df_queried.rename(columns={0:'Sch_Exchange', 'index':'MTU'})

    return df_queried


def achieve_hourly_granularity(df):

    """
    Process a DataFrame to achieve hourly granularity of flow data.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing flow data with columns 'MTU', 'From', 'To', and 'Flow_MW'.
                         Here, some MTU may be of 15 minutes granularity, and some may be of 1 hour granularity.

    Returns:
    pd.DataFrame: Processed DataFrame with hourly granularity, where the 'Flow_MW' values are averaged
                  for each hour between 'From' and 'To'.

    Notes:
    - The 'MTU' column is converted to UTC datetime.
    - New columns 'Hour' and 'Date' are created to store the hour and date information, respectively.
    - The DataFrame is then grouped by 'Date', 'Hour', 'From', and 'To' to calculate the average 'Flow_MW'.
    - The 'Date' column is adjusted to represent the midpoint of each hour by adding 30 minutes.
    - The 'Hour' column is dropped from the final result.

    """

    # Convert 'MTU' column to UTC datetime
    df['MTU'] = pd.to_datetime(df['MTU'], utc=True)

    # Create a new column 'Hour' to store the hour information
    df['Hour'] = df['MTU'].dt.hour

    # Create a new column 'Date' to store the date information
    df['Date'] = df['MTU'].dt.date

    # Group by 'Date', 'Hour', 'From', and 'To' and calculate the average flow value
    df = df.groupby(['Date', 'Hour', 'From', 'To'])['Sch_Exchange'].mean().reset_index()

    # Convert 'index' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    df['Date'] = df['Date'] + pd.to_timedelta(df['Hour'].astype(str) + ':00:00')
    df.drop(columns=['Hour'], inplace=True)

    df.rename(columns={'Date': 'MTU'}, inplace=True)

    return df

def main_border_queries(start_date, end_date):

    #####################
    ### MAPPING TABLE ###
    #####################

    NEIGHBOURS_Real_System = {
        'NL': ['NO_2', 'DK_1'],
        'DE_AT_LU': ['DK_1', 'DK_2', 'SE_4'],
        'GB': ['NO_2','DK_1'],
        'NO_2': ['DE_LU', 'DK_1', 'NL', 'NO_1', 'NO_5', 'GB'],
        'PL': ['SE_4'],
        'DK_1': ['DE_AT_LU', 'DE_LU', 'DK_2', 'NO_2', 'SE_3', 'NL','GB'],
        'LT': ['SE_4'],
        'SE_3': ['DK_1', 'FI', 'NO_1', 'SE_2', 'SE_4'],
        'NO_1': ['NO_2', 'NO_3', 'NO_5', 'SE_3'],
        'SE_4': ['DE_AT_LU', 'DE_LU', 'DK_2', 'LT', 'PL', 'SE_3'],
        'NO_5': ['NO_1', 'NO_2', 'NO_3'],
        'EE': ['FI'],
        'DK_2': ['DE_AT_LU', 'DE_LU', 'DK_1', 'SE_4'],
        'FI': ['EE', 'NO_4', 'RU', 'SE_1', 'SE_3'],
        'NO_4': ['SE_2', 'FI', 'NO_3', 'SE_1'],
        'SE_1': ['FI', 'NO_4', 'SE_2'],
        'SE_2': ['NO_3', 'NO_4', 'SE_1', 'SE_3'],
        'DE_LU': ['DK_1', 'DK_2', 'NO_2', 'SE_4'],
        'NO_3': ['NO_1', 'NO_4', 'NO_5', 'SE_2']
    }
    
    NTCs = queryWeekAheadCapacities(NEIGHBOURS_Real_System, start_date, end_date)
    NTCs = fromDailyToHourlyGranularity(NTCs, start_date, end_date)

    # Query cross-border scheduled exchanges from ENTSO-E
    SchExch = query_scheduled_exchanges_ENTSOE(NEIGHBOURS_Real_System, start_date, end_date)
    SchExch_H = achieve_hourly_granularity(SchExch)

    return NTCs, SchExch, SchExch_H




if __name__ == "__main__":
    
    import time

    start_time = time.time()
    
    start_date = pd.Timestamp('20240220 00:00:00', tz='Europe/Brussels').tz_convert('UTC')
    end_date = pd.Timestamp('20240220 02:00:00', tz='Europe/Brussels').tz_convert('UTC')

    NTCs, SchExch, SchExch_H = main_border_queries(start_date, end_date)

    # Putting the cross border exchange data in the right format
    NTCs_pivot = NTCs.copy(deep=True).reset_index()
    NTCs_pivot = NTCs_pivot.pivot_table(index=['MTU', 'biddingZoneFrom'], columns='biddingZoneTo', values='WeekAhead_NTC', aggfunc='first').reset_index()
    NTCs_pivot = NTCs_pivot.rename(columns={col: f'Cap_to_{col}' if col != 'MTU' and col != 'biddingZoneFrom' else col for col in NTCs_pivot.columns})
    NTCs_pivot = NTCs_pivot.fillna(0)
    NTCs_pivot.rename(columns={'biddingZoneFrom': 'From'}, inplace=True)

    # Scheduled Hourly Export
    SchExch_H_Ex = SchExch_H.copy(deep=True)
    SchExch_H_Ex = SchExch_H_Ex.pivot_table(index=['MTU', 'From'], columns='To', values='Sch_Exchange', aggfunc='first').reset_index()
    SchExch_H_Ex = SchExch_H_Ex.rename(columns={col: f'Ex_to_{col}' if col != 'MTU' and col != 'From' else col for col in SchExch_H_Ex.columns})
    SchExch_H_Ex = SchExch_H_Ex.fillna(0)

    # Scheduled Hourly Import
    SchExch_H_Imp = SchExch_H.copy(deep=True)
    SchExch_H_Imp['From'], SchExch_H_Imp['To'] = SchExch_H_Imp['To'], SchExch_H_Imp['From']
    SchExch_H_Imp = SchExch_H_Imp.pivot_table(index=['MTU', 'From'], columns='To', values='Sch_Exchange', aggfunc='first').reset_index()
    SchExch_H_Imp = SchExch_H_Imp.rename(columns={col: f'Imp_from_{col}' if col != 'MTU' and col != 'From' else col for col in SchExch_H_Imp.columns})
    SchExch_H_Imp = SchExch_H_Imp.fillna(0)

    # Scheduled exchanges merge
    Border_data = SchExch_H_Ex.merge(SchExch_H_Imp, on=['MTU', 'From'], how='left')
    Border_data['MTU'] = pd.to_datetime(Border_data['MTU']).dt.tz_localize('UTC')
    Border_data = Border_data.merge(NTCs_pivot, on=['MTU', 'From'], how='left')

    end_time = time.time()

    # convert these to minutes and seconds
    minutes, seconds = divmod(end_time - start_time, 60)
    print(f'The processing time was {minutes:.0f} minutes and {seconds:.0f} seconds.')

# %%
