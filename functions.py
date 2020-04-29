
import os
import pandas as pd
import numpy as np
import configparser as cp

from datetime import date


def get_ecdc_data(overwrite = False):
    '''
    Gets latest report on covid cases from ECDC url.
    Checks and clears the data, adds analytical features form LUTs.
    

    Parameters
    ----------
    overwrite : bool
        If to process again today's data.

    Returns
    ----------
    pd.DataFrame
        
    '''

    # Check if today's data has already been processed.
    # If the data is found (and overwrite is not forces) return it.
    today = date.today().strftime("%Y_%m_%d")
    processed_name = 'ecdc_'+today+'.csv'

    if (processed_name in os.listdir('./data')) & (not overwrite):
        df = pd.read_csv('./data/' + processed_name)
        return df

    # -- 01 get data ------------------------------------
    
    #get COVID data url from config file
    config = cp.ConfigParser()
    config.read_file(open('config.cfg'))
    data_url = config.get('ECDC', "URL")

    #import raw data frame
    df = pd.read_csv(data_url)

    #check if data frame has all required columns
    required = ['dateRep','cases','countriesAndTerritories','popData2018']
    raw_names = list(df.columns)
    missing = list(set(required).difference(raw_names))

    assert not missing, \
        'Reqired column(s): {} are missing.'.format(missing)

    #import and add geo data from LUT
    geo_dict = config.get("VARIA", "GEO_DICT")
    cont_dict = pd.read_csv(geo_dict, na_filter = False)

    df = df.merge(cont_dict, on='countriesAndTerritories', how = 'left')

    # -- 02 clear data ------------------------------------

    #clear column names
    df=df[['countriesAndTerritories', 'Cont', 'Cont_Det', 'popData2018', 'dateRep','cases','deaths']]
    df.columns = ['Country', 'Cont', 'Cont_Det', 'Pop', 'Date', 'Cases', 'Deaths']

    #format date
    df['Date']=df['Date'].apply(lambda x: pd.datetime.strptime(x, '%d/%m/%Y'))

    # clear country name
    df['Country'] = df['Country'].str.replace('_', ' ') 
    df['Country'] = df['Country'].str.title()
    df['Country'] = df['Country'].str.replace(' ', '_') 
    df['Country'] = df['Country'].str.replace('United_States_Of_America', 'USA') 
    df['Country'] = df['Country'].str.replace('United_Kingdom', 'UK')
    df['Country'] = df['Country'].str.replace('United_Republic_Of_Tanzania', 'Tanzania')
    df['Country'] = df['Country'].str.replace('Democratic_Republic_Of_The_Congo', 'Congo')
    df['Country'] = df['Country'].str.replace('Cases_On_An_International_Conveyance_Japan', 'Japan')

    #clear and format pop
    df['Pop']=df['Pop'].fillna(0)
    df['Pop']=round(df['Pop'].astype(int)/1000000.0,2)

    #sort by country and date
    df.sort_values(by=['Country', 'Date'], inplace=True)

    # -- 03 add some aggregates and flags ------------------------------------

    df['Cases_cum'] = df.groupby(['Country'])['Cases'].cumsum()
    df['Deaths_cum'] = df.groupby(['Country'])['Deaths'].cumsum()
    df['Deaths_Ratio_cum'] = df['Deaths_cum']/df['Cases_cum']
    df['Deaths_per1MPop'] = round(df['Deaths_cum']/df['Pop'], 2)

    #filter out days pre case
    df = df[df['Cases_cum']+df['Deaths_cum']>0]

    df['has_10_deaths'] = (df['Deaths_cum']>=10).astype(int)
    df['has_1_death_per5M'] = (df['Deaths_per1MPop']>=0.2).astype(int)
    df['days_since_1_per5M'] = df.groupby(['Country'])['has_1_death_per5M'].cumsum()

    df.drop(['has_10_deaths','has_1_death_per5M'], axis=1, inplace=True)

    # -- 04 write and return processed data ------------------------------------

    df.to_csv('./data/'+processed_name, index = False)

    return df




