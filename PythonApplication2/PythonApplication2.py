# -*- coding: cp949 -*-


import pandas as pd

import chardet

# dataFile.csv: Number of atopic dermatitis treatments
with open('./dataFile.csv', 'rb') as file:
    result = chardet.detect(file.read())

# dataFile2.csv: Average, Maximum, Minimum temperature of Korea. Average and daily temperature range (Max-Min) will be used.
with open('./dataFile2.csv', 'rb') as file2:
    result2 = chardet.detect(file2.read())

treatment_df = pd.read_csv('./dataFile.csv', header = 0, encoding = result['encoding']) 

temperature_df = pd.read_csv('./dataFile2.csv', header = 6, encoding = result['encoding'])

# change date to datetime.
treatment_df['요양개시일'] = pd.to_datetime(treatment_df['요양개시일'])

temperature_df['날짜'] = pd.to_datetime(temperature_df['날짜'])

# add daily temperature range.
temperature_df['일교차'] = temperature_df['최고기온(℃)'] - temperature_df['최저기온(℃)']

# Put the data together on a weekly basis.
treatment_df['주간별진료'] = treatment_df['요양개시일'].dt.to_period('W')

temperature_df['주간별기온'] = temperature_df['날짜'].dt.to_period('W')

# get add of all tretment count for weekly.
weekly_Treatment_Data = treatment_df.groupby('주간별진료')['진료에피소드건수'].sum()

# get mean of average temperature and temperaturen range. 
weekly_Temperature_Data = temperature_df.groupby('주간별기온').agg({
    '평균기온(℃)': 'mean',
    '일교차': 'mean'
}).reset_index()

print(weekly_Treatment_Data)

print(weekly_Temperature_Data)
