
# -- 00 load libs ------------------------------------

import pandas as pd
import numpy as np


# -- 01 get data ------------------------------------
dp = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')

month_day = '0318'

df=pd.read_csv('./data/ecdc_20'+month_day+'.csv', parse_dates=[0], date_parser=dp)
df.columns
df

#get current date
df.groupby(['DateRep'])['Cases'].sum().sort_index(ascending=False)


# clean country name
df.rename(columns={"Countries and territories": "Country"}, inplace=True)
df['Country'] = df['Country'].str.replace('_', ' ') 
df['Country'] = df['Country'].str.title()
df['Country'] = df['Country'].str.replace(' ', '_') 
df['Country'] = df['Country'].str.replace('United_States_Of_America', 'USA') 
df['Country'] = df['Country'].str.replace('United_Kingdom', 'UK')
df['Country'] = df['Country'].str.replace('United_Republic_Of_Tanzania', 'Tanzania')
df['Country'] = df['Country'].str.replace('Democratic_Republic_Of_The_Congo', 'Congo')


df.rename(columns={'DateRep':'Date'}, inplace=True)

df = df[['GeoId', 'Country', 'Date', 'Cases', 'Deaths']]

#sort by country and date
df.sort_values(by=['Country', 'Date'], inplace=True)


# -- 02 add some aggregates and flags ------------------------------------

df['Cases_cum'] = df.groupby(['Country'])['Cases'].cumsum()
df['Deaths_cum'] = df.groupby(['Country'])['Deaths'].cumsum()
df['Deaths_Ratio_cum'] = df['Deaths_cum']/df['Cases_cum']

#filter out days pre case

df = df[df['Cases_cum']+df['Deaths_cum']>0]

#df['has_200_cases'] = (df['Cases_cum']>=200).astype(int)
df['has_1k_cases'] = (df['Cases_cum']>=1000).astype(int)
df['has_10_deaths'] = (df['Deaths_cum']>=10).astype(int)

#df['days_since_200_cases'] = df.groupby(['Country'])['has_200_cases'].cumsum()
df['days_since_1k_cases'] = df.groupby(['Country'])['has_1k_cases'].cumsum()
df['days_since_10_deaths'] = df.groupby(['Country'])['has_10_deaths'].cumsum()

df.drop(['has_1k_cases','has_10_deaths'], axis=1, inplace=True)






# -- 03 get some statistics ------------------------------------

df[df['days_since_10_deaths']>0].head(100)

df[df['Country']=='Poland'].head(100)
df[df['Country']=='Italy'].head(100)

df[df['Date']=='2020-03-18'].sort_values(by=['Cases'], ascending=False)
df[df['Date']=='2020-03-18'].sort_values(by=['Deaths'], ascending=False)

df.groupby(['Country'])['Cases'].sum().sort_values(ascending=False).head(50)
df.groupby(['Country'])['Deaths'].sum().sort_values(ascending=False).head(30)

df['Country'].value_counts()

# -- 04 make some plots (Plotly) ------------------------------------

import plotly as pl
import plotly.express as px

px.line(
    df[(df['days_since_1k_cases']>0) 
        # & (df['Country']!='China')
        ]
    ,x='days_since_1k_cases'
    ,y='Cases_cum'
    ,color='Country'
    ,log_y=True
).show()

px.line(
    df[(df['days_since_10_deaths']>0) 
        & (df['Country']!='China')
        ]
    ,x='days_since_10_deaths'
    ,y='Deaths_cum'
    ,color='Country'
    ,log_y=True
).show()

px.line(
    df[(df['days_since_10_deaths']>0) 
        & (df['Country']!='China')
        ]
    ,x='days_since_10_deaths'
    ,y='Deaths_Ratio_cum'
    ,color='Country'
    # ,log_y=True
).show()



df['Deaths_cum_sh7'] = df['Deaths_cum'].shift(-4)
df['Deaths_Ratio_cum_sh7']=df['Deaths_cum_sh7']/df['Cases_cum']


px.line(
    df[(df['days_since_10_deaths']>0) 
        & (df['Country']!='China')
        & (df['Deaths_Ratio_cum_sh7']>0)
        ]
    ,x='days_since_10_deaths'
    ,y='Deaths_Ratio_cum_sh7'
    ,color='Country'
    # ,log_y=True
).show()




# -- 05 cases and deth cross-corelation ------------------------------------

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

cntr = 'Italy'
c = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Cases_cum']
d = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Deaths_cum']
c0 = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Cases']
d0 = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Deaths']

cntr = 'China'
c = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Cases_cum']
d = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Deaths_cum']
c0 = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Cases']
d0 = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Deaths']

cntr = 'Iran'
c = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Cases_cum']
d = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Deaths_cum']
c0 = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Cases']
d0 = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Deaths']

cntr = 'South_Korea'
c = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Cases_cum']
d = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Deaths_cum']
c0 = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Cases']
d0 = df[(df['Country']==cntr) & (df['days_since_10_deaths']>0)]['Deaths']


px.line(x=range(0,20), y=list(map(lambda x: crosscorr(d, c, lag=x), range(0,20)))).show()
px.line(x=range(0,20), y=list(map(lambda x: crosscorr(d0, c0, lag=x), range(0,20)))).show()
px.line(x=range(0,20), y=list(map(lambda x: crosscorr(d0, c, lag=x), range(0,20)))).show()



# -- 04 make some plots (Seaborn) ------------------------------------


import seaborn as sns
sns.set(style="whitegrid")

sns.lineplot(
    data=df[df['Country']=='Poland']
    ,x='Date'
    ,y='Cases')

p = sns.lineplot(
    data=df[(df['days_since_10_deaths']>0) & (df['Country']!='China')]
    ,x='days_since_10_deaths'
    ,y='Deaths_cum'
    ,hue='Country'
    )
p.set(yscale="log")


sns.lineplot()

sns.lineplot(df['days_since_10_deaths'>0]['Deaths'], groupby=df['Country'])

rs = np.random.RandomState(365)
values = rs.randn(365, 4).cumsum(axis=0)
dates = pd.date_range("1 1 2016", periods=365, freq="D")
data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
data = data.rolling(7).mean()

sns.lineplot(data=data, palette="tab10", linewidth=2.5)
