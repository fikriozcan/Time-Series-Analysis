# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:22:41 2019

@author: ASUSNB
"""


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
% matplotlib inline
df = pd.read_csv('Data Exercise_.csv',index_col = 'Date',parse_dates=True)

df.info()

# Counting unique variables of each column
from collections import Counter
Counter(df['Vendor_Name'])

Counter(df['Product'])

Counter(df['Country'])

## replacing space with _ in order to read column names correctly
df.columns = df.columns.str.replace(' ', '_')

# removing na values from dataset 
df = df.dropna()

# replacing space with _
df['_Fees_'] = df._Fees_.str.replace(' ', '_')

# replacing string values of Fees with zero
df._Fees_[df._Fees_ == '_$-___'] = 0

# converting fees column from string to numeric         
df["_Fees_"] = pd.to_numeric(df["_Fees_"])

# same process had been impelemented for Gross Bookings variable as well. 
df['_Gross_Bookings_'] = df._Gross_Bookings_.str.replace(' ', '_')

df._Gross_Bookings_[df._Gross_Bookings_ == '_$-___'] = 0

df["_Gross_Bookings_"] = pd.to_numeric(df["_Gross_Bookings_"])

##################

# getting index column as Date column and using group by function 
df.reset_index(inplace=True)

# since there are too much zeros in our data, the number of instances varies in each month, mean
# has been used to create subset
ax = df.groupby('Date')['_Gross_Bookings_'].mean().plot()

# creating dataframe called df_1 which is grouped by date. 
df_1 = df.groupby('Date')['_Gross_Bookings_','_Fees_'].mean()

# using moving average 3  to see how moving average fits in gross bookings 
df_1['moving_avg_3']= df_1.rolling(window=3).mean()['_Gross_Bookings_']

## visualising both gross booking and moving average. Plot indicates that, MA(3) fits pretty well but lag could be observed easily. 
ax = df_1[['_Gross_Bookings_','moving_avg_3']].plot(figsize=(12,5), title = 'Comparison btw Gross Booking & Moving Average' ,legend = True)
ax.autoscale(axis='both' , tight = True)
ax.set(xlabel = 'Date' , ylabel = '$')
plt.legend(['Gross Bookings','Moving Average (3 Months)'])
plt.show()

## visualising moving average 3 seperately 
df_1.rolling(window=3).mean()['_Gross_Bookings_'].plot()

## visualising distribution of fees over months. 
df.groupby('Date')['_Fees_'].mean().plot()


##     
import statsmodels.api as sm
cycle, trend = sm.tsa.filters.hpfilter(df_1._Gross_Bookings_, 129600)

df_1['trend'] = trend
df_1['cycle'] = cycle

## Seperating trend and cyclic part of the variable and visiualising all three lines in the same graph
ax = df_1[['_Gross_Bookings_','trend','cycle']].plot(figsize=(12,5), title = 'title' ,legend = True)
ax.autoscale(axis='both' , tight = True)
ax.set(xlabel = 'Date' , ylabel = '$')
plt.legend(['Gross Bookings','Trend','Cycle'])
plt.show()


### Predictive model by using Simple Exponential Smoothing  
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
span = 12 
alpha = 2 / (span + 1)

df_1['EWMA12'] = df_1['_Gross_Bookings_'].ewm(alpha = alpha, adjust =  False).mean()
df_1.head()

model = SimpleExpSmoothing(df_1['_Gross_Bookings_'])
fitted_model = model.fit(smoothing_level=alpha,optimized= False)
df_1['ses12'] = fitted_model.fittedvalues.shift(-1)


test_predictions = fitted_model.forecast(1)
test_predictions




