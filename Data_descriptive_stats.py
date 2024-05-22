#%%
#load csv-files from data folder

import pandas as pd
import numpy as np
import os

data_frame_lists = ['day_ahead_prices', 'load', 'load_forecast', 'load_and_forecast', 'generation_forecast', 'wind_and_solar_forecast'] #  'crossborder_flows', 'scheduled_exchanges', 'net_transfer_capacity', 'aggregate_water_reservoirs_hydro_storage',

work_dir = os.getcwd()

# import data using os library
data = {}
for data_frame in data_frame_lists:
    data[data_frame] = pd.read_csv(os.path.join(work_dir + os.sep + r'data\2023', data_frame + '.csv'))
    
# make dataframe data_df from the dict for each data_frame
day_ahead_prices = data['day_ahead_prices']
day_ahead_prices.rename(columns={'0': 'DA-price [EUR/MWh]'}, inplace=True)

load_and_forecast = data['load_and_forecast']
generation_forecast = data['generation_forecast']
generation_forecast.rename(columns={'Actual Aggregated': 'Forecasted Generation'}, inplace=True)
wind_and_solar_forecast = data['wind_and_solar_forecast']

#merge dataframes based on the Unnamed: 0 column and country_code column
data_df = pd.merge(day_ahead_prices, load_and_forecast, on=['Unnamed: 0', 'country_code'])
data_df = pd.merge(data_df, generation_forecast, on=['Unnamed: 0', 'country_code'])
data_df = pd.merge(data_df, wind_and_solar_forecast, on=['Unnamed: 0', 'country_code'])

#rename Unnamed: 0 to Timestamp
data_df.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
#make Timestamp to datatime format
data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp'])


# %%
import matplotlib.pyplot as plt

# Get unique country codes
unique_country_codes = data_df['country_code'].unique()

# Create subplots for each country code in a 3x4 grid
fig, axs = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)

# Flatten axs array for easier iteration
axs = axs.flatten()

# Plot DA-prices with the country_code as labels
for i, country_code in enumerate(unique_country_codes):
    data_df_country = data_df[data_df['country_code'] == country_code]
    axs[i].plot(data_df_country['Timestamp'], data_df_country['DA-price [EUR/MWh]'])
    axs[i].set_ylabel('DA-price [EUR/MWh]')
    axs[i].set_title(f'Country Code: {country_code}')
    axs[i].tick_params(axis='x', rotation=45)


# Set common x-label
plt.xlabel('Timestamp')
# make overall title
plt.suptitle('Day-ahead prices for different bidding zones')
#rotate x-labels for all subplots


# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

#%%
#make similar plots for the other dataframes
#load
fig, axs = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
axs = axs.flatten()
for i, country_code in enumerate(unique_country_codes):
    data_df_country = data_df[data_df['country_code'] == country_code]
    axs[i].plot(data_df_country['Timestamp'], data_df_country['Forecasted Load'])
    axs[i].set_ylabel('Load [MW]')
    axs[i].set_title(f'Country Code: {country_code}')
    axs[i].tick_params(axis='x', rotation=45)

plt.xlabel('Timestamp')
plt.suptitle('Forecasted Load for different bidding zones')

plt.tight_layout()
plt.show()

#%%
#generation forecast
fig, axs = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
axs = axs.flatten()
for i, country_code in enumerate(unique_country_codes):
    data_df_country = data_df[data_df['country_code'] == country_code]
    axs[i].plot(data_df_country['Timestamp'], data_df_country['Forecasted Generation'])
    axs[i].set_ylabel('Generation [MW]')
    axs[i].set_title(f'Country Code: {country_code}')
    axs[i].tick_params(axis='x', rotation=45)
    
plt.xlabel('Timestamp')
plt.suptitle('Forecasted Generation for different bidding zones')

plt.tight_layout()
plt.show()

#%%
#wind and solar forecast
fig, axs = plt.subplots(3, 4, figsize=(15, 10), sharex=True, sharey=True)
axs = axs.flatten()
for i, country_code in enumerate(unique_country_codes):
    data_df_country = data_df[data_df['country_code'] == country_code]
    axs[i].plot(data_df_country['Timestamp'], data_df_country['Wind Offshore'], label='Wind offshore')
    axs[i].plot(data_df_country['Timestamp'], data_df_country['Solar'], label='Solar')
    axs[i].plot(data_df_country['Timestamp'], data_df_country['Wind Onshore'], label='Wind onshore')
    axs[i].set_ylabel('Generation [MW]')
    axs[i].set_title(f'Country Code: {country_code}')
    axs[i].tick_params(axis='x', rotation=45)
    axs[i].legend()
    
plt.xlabel('Timestamp')
plt.suptitle('Forecasted Wind and Solar Generation for different bidding zones')

plt.tight_layout()
plt.show()


# %%
