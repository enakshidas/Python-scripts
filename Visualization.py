# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:59:38 2019

@author: Enakshi
"""

import pandas as pd
import numpy as np
import calendar
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

df_file= pd.read_csv('Viz Data - SIPROD.csv')  
df_file[['month','day','year']] = df_file.date.str.split('/', expand=True)


df_file['month']=pd.to_numeric(df_file['month'])

df_file['day']=pd.to_numeric(df_file['day'])
df_file =df_file[df_file['day'] < 32]

df_file['year']=pd.to_numeric(df_file['year'])
df_file['year'] = df_file['year']

df_file['Date'] = df_file.apply(lambda row: datetime(
                              row['year'], row['month'], row['day']), axis=1)

df_file['Date'] = pd.to_datetime(df_file['Date'])
df_file['Date_Occured'] = df_file['Date'].dt.date

df_file.drop(['month','year','day', 'date','Date'], axis=1, inplace=True)

writer = pd.ExcelWriter('visualized_Cleaned.xlsx', engine='xlsxwriter')

df_file.to_excel(writer, sheet_name='Sheet1')

writer.save()



#Visualization for Issues VS Count

colorFreqTable= df_file['issue'].value_counts()
list(colorFreqTable.index)
colorFreqTable.values
# get all the issue names from our frequency plot & save them for later
labelNames = list(colorFreqTable.index)
# generate a list of numbers
positionsForBars = list(range(len(labelNames)))

#pass the names and counts to the bar function

plt.bar(positionsForBars,colorFreqTable.values) # plot our bars
plt.xticks(positionsForBars,labelNames) # add lables
plt.title("Issue VS Count", fontweight="bold",fontsize=21)


#Visualization for count vs date with respect to Issues

visualize_df = (df_file.reset_index().groupby(['Date_Occured','issue'], as_index=False).count().rename(columns={'index':'count'}))
fig, ax = plt.subplots()
plt.title("Visualization for count vs date")
for key, data in visualize_df.groupby('issue'):data.plot(x='Date_Occured', y='count', ax=ax, label=key)