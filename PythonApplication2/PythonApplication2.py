# -*- coding: cp949 -*-


from imp import acquire_lock
from re import A
import pandas as pd

import chardet

import matplotlib.pyplot as plt

# dataFile.csv: Number of atopic dermatitis treatments
with open('./dataFile.csv', 'rb') as file:
    result = chardet.detect(file.read())

# dataFile2.csv: Average, Maximum, Minimum temperature of Korea. Average and daily temperature range (Max-Min) will be used.
with open('./dataFile2.csv', 'rb') as file2:
    result2 = chardet.detect(file2.read())

# datafile 3,4: Air pollution data(Seoul)
with open('./dataFile3.csv', 'rb') as file3:
    result3 = chardet.detect(file3.read())
with open('./dataFile4.csv', 'rb') as file4:
    result4 = chardet.detect(file4.read())


treatment_df = pd.read_csv('./dataFile.csv', header = 0, encoding = result['encoding']) 

temperature_df = pd.read_csv('./dataFile2.csv', header = 6, encoding = result2['encoding'])

air_df_2018 = pd.read_csv('./dataFile3.csv', header = 0, encoding = result3['encoding'])
air_df_2019 = pd.read_csv('./dataFile4.csv', header = 0, encoding = result4['encoding'])

air_df_2018['�����Ͻ�'] = pd.to_datetime(air_df_2018['�����Ͻ�'], format='%Y%m%d')
air_df_2019['�����Ͻ�'] = pd.to_datetime(air_df_2019['�����Ͻ�'], format='%Y%m%d')

pm_columns_2018 = [col for col in air_df_2018.columns if '�̼�����' in col]  
pm_columns_2019 = [col for col in air_df_2019.columns if '�̼�����' in col]  

air_df_2018['average_pm10'] = air_df_2018[pm_columns_2018].mean(axis=1, skipna=True)
air_df_2019['average_pm10'] = air_df_2019[pm_columns_2019].mean(axis=1, skipna=True)

pm_columns_2018_2 = [col for col in air_df_2018.columns if '�ʹ̼�' in col]  
pm_columns_2019_2 = [col for col in air_df_2019.columns if '�ʹ̼�' in col]  

air_df_2018['average_pm25'] = air_df_2018[pm_columns_2018_2].mean(axis=1, skipna=True)
air_df_2019['average_pm25'] = air_df_2019[pm_columns_2019_2].mean(axis=1, skipna=True)


start_date = '2018-02-01'
end_date = '2019-07-31'

df_2018_filtered = air_df_2018[(air_df_2018['�����Ͻ�'] >= start_date) & (air_df_2018['�����Ͻ�'] <= end_date)]
df_2019_filtered = air_df_2019[(air_df_2019['�����Ͻ�'] >= start_date) & (air_df_2019['�����Ͻ�'] <= end_date)]

air_df_filtered = pd.concat([df_2018_filtered[['�����Ͻ�', 'average_pm10', 'average_pm25']], df_2019_filtered[['�����Ͻ�', 'average_pm10', 'average_pm25']]])

air_df_filtered = air_df_filtered.drop_duplicates()

air_df_grouped = air_df_filtered.groupby('�����Ͻ�').mean()
# change date to datetime.
treatment_df['��簳����'] = pd.to_datetime(treatment_df['��簳����'])


temperature_df['��¥'] = pd.to_datetime(temperature_df['��¥'])


# add a new column to calculate the difference in average temperature and daily temperature change.
temperature_df['������ձ��'] =  temperature_df['��ձ��(��)'].shift(1)

temperature_df['��ձ����'] = (temperature_df['��ձ��(��)'] - temperature_df['������ձ��']).abs()

temperature_df['�ϱ���'] = temperature_df['�ְ���(��)'] - temperature_df['�������(��)']

# Put the data together on a weekly basis.
treatment_df['��簳����'] = treatment_df['��簳����'].dt.to_period('W')

treatment_df['�ְ�������'] = treatment_df.groupby('��簳����')['���ῡ�Ǽҵ�Ǽ�'].transform('sum')

treatment_df = treatment_df.drop(columns=['���ῡ�Ǽҵ�Ǽ�'])

treatment_df = treatment_df.drop_duplicates()


air_df_grouped = air_df_grouped.reset_index()
air_df_grouped['�����Ͻ�'] = pd.to_datetime(air_df_grouped['�����Ͻ�'])
air_df_grouped['�����Ͻ�'] = air_df_grouped['�����Ͻ�'].dt.to_period('W')

air_df_grouped['�ְ��̼�����'] = air_df_grouped.groupby('�����Ͻ�')['average_pm10'].transform('mean')

air_df_grouped['�ְ��ʹ̼�����'] = air_df_grouped.groupby('�����Ͻ�')['average_pm25'].transform('mean')
air_df_grouped = air_df_grouped.drop('average_pm10', axis = 1)
air_df_grouped = air_df_grouped.drop('average_pm25', axis = 1)

temperature_df['��¥'] = temperature_df['��¥'].dt.to_period('W')
air_df_grouped = air_df_grouped.drop_duplicates()

air_df_grouped = air_df_grouped.reset_index(drop=True)
print(air_df_grouped)


temperature_df['�ְ���ձ����'] = temperature_df.groupby('��¥')['��ձ����'].transform('mean')

temperature_df['�ְ��ϱ������'] = temperature_df.groupby('��¥')['�ϱ���'].transform('mean')

# New dataframe for calculated.
temperature_range_df = pd.DataFrame(columns = ['�ְ���', '�ְ��ϱ������', '���ֺ񱳿µ���'])


# fill columns

temperature_range_df['�ְ���'] = temperature_df['��¥']

temperature_range_df['�ְ��ϱ������'] = temperature_df['�ְ��ϱ������']

temperature_range_df['���ֺ񱳿µ���'] = temperature_df['�ְ���ձ����'] 

temperature_range_df = temperature_range_df.drop_duplicates()





# Change column name.
treatment_df = treatment_df.rename(columns={'��簳����': '�ְ���'})

# Concatenate two dataframes.
combined_df = pd.concat([treatment_df, temperature_range_df], axis = 1)

combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]


combined_df = combined_df.reset_index(drop=True)

print(combined_df)

combined_df = pd.concat([combined_df, air_df_grouped], axis = 1)



corr1 = combined_df['�ְ�������'].corr(combined_df['�ְ��ϱ������'])
corr2 = combined_df['�ְ�������'].corr(combined_df['���ֺ񱳿µ���'])
corr3 = combined_df['�ְ�������'].corr(combined_df['�ְ��̼�����'])
corr4 = combined_df['�ְ�������'].corr(combined_df['�ְ��ʹ̼�����'])

print(corr1)
print(corr2)
print(corr3)
print(corr4)