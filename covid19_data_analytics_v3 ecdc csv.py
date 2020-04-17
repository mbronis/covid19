
# -- 01 get data ------------------------------------

import pandas as pd
import numpy as np

data_url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
df = pd.read_csv(data_url)

#add continents info
cont_dict = pd.read_csv('./data/country_continent_dic.csv', na_filter = False)
df = df.merge(cont_dict, on='countriesAndTerritories', how = 'left')

df['Date']=df['dateRep'].apply(lambda x: pd.datetime.strptime(x, '%d/%m/%Y'))
df=df[['countriesAndTerritories', 'Cont', 'Cont_Det', 'popData2018', 'Date','cases','deaths']]
df.columns = ['Country', 'Cont', 'Cont_Det', 'Pop', 'Date', 'Cases', 'Deaths']


# clean country name
df['Country'] = df['Country'].str.replace('_', ' ') 
df['Country'] = df['Country'].str.title()
df['Country'] = df['Country'].str.replace(' ', '_') 
df['Country'] = df['Country'].str.replace('United_States_Of_America', 'USA') 
df['Country'] = df['Country'].str.replace('United_Kingdom', 'UK')
df['Country'] = df['Country'].str.replace('United_Republic_Of_Tanzania', 'Tanzania')
df['Country'] = df['Country'].str.replace('Democratic_Republic_Of_The_Congo', 'Congo')
df['Country'] = df['Country'].str.replace('Cases_On_An_International_Conveyance_Japan', 'Japan')

#clean and format pop
df['Pop']=df['Pop'].fillna(0)
df['Pop']=round(df['Pop'].astype(int)/1000000.0,2)

#sort by country and date
df.sort_values(by=['Country', 'Date'], inplace=True)

#get current date
month = pd.datetime.date(max(df['Date'])).month
day = pd.datetime.date(max(df['Date'])).day
month_day = str(month).zfill(2) + '-' + str(day).zfill(2)

# -- 02 add some aggregates and flags ------------------------------------

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

#europe only
eu = df[df['Cont']=='EU']

# -- 03 get some statistics ------------------------------------

#PL
df[df['Country']=='Poland'].sort_values('Date', ascending=False).head(50)

# world #deaths
df[df['Date']=='2020-'+month_day].sort_values(by=['Deaths_cum'], ascending=False).head(50)
# world 10M+
df[(df['Date']=='2020-'+month_day) & (df['Pop']>10)].sort_values(by=['Deaths_per1MPop'], ascending=False).head(50)
# eu 5M+
eu[(eu['Date']=='2020-'+month_day) & (eu['Pop']>5)].sort_values(by=['Deaths_per1MPop'], ascending=False).head(50)


d=23

# eu 5M+
df[(df['days_since_1_per5M']==d)&(df['Pop']>5)&(df['Cont']=='EU')].sort_values(by=['Deaths_per1MPop'], ascending=False).head(100)
# world 10M+
df[(df['days_since_1_per5M']==d)&(df['Pop']>10)].sort_values(by=['Deaths_per1MPop'], ascending=False).head(100)

df[df['Country']=='Greece'].sort_values('Date', ascending=False).head(20)
df[df['Country']=='Italy'].sort_values('Date', ascending=False).head(40)
df[df['Country']=='Spain'].sort_values('Date', ascending=False).head(20)
df[df['Country']=='USA'].sort_values('Date', ascending=False).head(20)
df[df['Country']=='Germany'].sort_values('Date', ascending=False).head(20)


# -- 04 make some plots (Plotly) ------------------------------------

import plotly as pl
import plotly.express as px

px.line(
    df[(df['days_since_1_per5M']>0) 
        & (df['Country']!='Japan')
        & (df['Pop']>37)
        & (df['Cont'].isin(['EU', 'NA', 'AS']))
        ]
    ,x='days_since_1_per5M'
    ,y='Deaths_per1MPop'
    ,color='Country'
    ,log_y=True
).show()

px.line(
    df[(df['days_since_1_per5M']>0) 
        # & (df['Country']!='China')
        & (df['Pop']>10)
        & (df['Cont']=='EU')
        ]
    ,x='days_since_1_per5M'
    ,y='Deaths_per1MPop'
    ,color='Country'
    ,log_y=True
).show()

px.line(
    df[(df['days_since_1_per5M']>0) 
        # & (df['Country']!='China')
        & (df['Pop']>10)
        & (df['Cont']=='EU')
        ]
    ,x='days_since_1_per5M'
    ,y='Deaths_per1MPop'
    ,color='Country'
    ,log_y=False
).show()


px.line(
    df[(df['days_since_1_per5M']>0) 
        # & (df['Country']!='China')
        & (df['Pop']>10)
        & (df['Cont']=='EU')
        ]
    ,x='days_since_1_per5M'
    ,y='Deaths_per1MPop'
    ,color='Country'
    ,log_y=False
).show()




px.line(
    df[(df['days_since_10_deaths']>0) 
        # & (df['Country']!='China')
        & (df['Pop']>35)
        & (df['Cont'].isin(['EU', 'NA', 'AS']))
        ]
    ,x='days_since_10_deaths'
    ,y='Deaths_cum'
    ,color='Country'
    ,log_y=True
).show()

px.line(
    df[(df['days_since_1_per5M']>0) 
        # & (df['Country']!='China')
        & (df['Pop']>30)
        & (df['Cont']=='EU')
        ]
    ,x='days_since_1_per5M'
    ,y='Deaths_per1MPop'
    ,color='Country'
    ,log_y=False
).show()

px.line(
    df[(df['days_since_10_deaths']>0) 
        # & (df['Country']!='China')
        & (df['Pop']>10)
        & (df['Cont']=='EU')
        ]
    ,x='days_since_10_deaths'
    ,y='Deaths_cum'
    ,color='Country'
    ,log_y=True
).show()

px.line(
    df[(df['days_since_10_deaths']>0) 
        # & (df['Country']!='China')
        & (df['Pop']>60)
        # & (df['Cont']=='EU')
        ]
    ,x='days_since_10_deaths'
    ,y='Deaths_per1MPop'
    ,color='Country'
    ,log_y=True
).show()

px.line(
    df[(df['days_since_1_per5M']>0) 
        # & (df['Country']!='China')
        & (df['Pop']>10)
        & (df['Cont']=='EU')
        ]
    ,x='days_since_1_per5M'
    ,y='Deaths_per1MPop'
    ,color='Country'
    ,log_y=True
).show()



d=5
l2=df[(df['days_since_10_deaths']==d) & (df['Deaths_cum']<30)]['Country'].unique()
px.line(
    df[(df['Country'].isin(l2)) 
        & (df['Country']!='China')
        & (df['Country']!='Italy')
        & (df['Country']!='USA')
        ]
    ,x='days_since_10_deaths'
    ,y='Deaths_cum'
    ,color='Country'
    ,log_y=False
).show()


l3=df[(df['Date']=='2020-'+month_day) & (df['days_since_10_deaths']>21)]['Country'].unique()
l3

l3=df[(df['Date']=='2020-'+month_day) & (df['days_since_10_deaths']>14)]['Country'].unique()
px.line(
    df[(df['days_since_10_deaths']>0)
        &(df['Country'].isin(l3)) 
        & (df['Country']!='China')
        # & (df['Country']!='Italy')
        # & (df['Country']!='USA')
        ]
    ,x='days_since_10_deaths'
    ,y='Deaths_cum'
    ,color='Country'
    ,log_y=True
).show()

px.line(
    df[(df['days_since_1k_cases']>0) 
        # & (df['days_since_10_deaths']>0) 
        & (df['Country']!='China')
        & (df['Country']!='Japan')
        ]
    ,x='days_since_1k_cases'
    ,y='Cases_cum'
    ,color='Country'
    ,log_y=True
).show()





c_10d=list(df[df['days_since_10_deaths']>10]['Country'].unique())
px.line(
    df[(df['Country'].isin(c_10d)) 
        & (df['Country']!='China')
        ]
    ,x='days_since_10_deaths'
    ,y='Deaths_cum'
    ,color='Country'
    ,log_y=True
).show()

px.line(
    df[(df['days_since_10_deaths']>0) 
        # & (df['Country']!='China')
        ]
    ,x='days_since_10_deaths'
    ,y='Deaths_cum'
    ,color='Country'
    ,log_y=True
).show()

px.line(
    df[(df['days_since_10_deaths']>0) 
        # & (df['Country']!='China')
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
